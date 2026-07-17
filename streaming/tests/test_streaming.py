"""当前两层 U-Net 的结构、导出和数值测试。"""
import numpy as np
import onnx
import onnxruntime as ort
import torch

from streaming.converters.convert_unet import convert
from streaming.models.unet import DeBlock, EnBlock, UNetAttn


def export(model, dummy):
    torch.onnx.export(
        model, dummy, "unet.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "time"},
                      "output": {0: "batch", 2: "time"}},
        opset_version=13, dynamo=False,
    )


def main():
    torch.manual_seed(0)
    model = UNetAttn().eval()
    dummy = torch.randn(1, 1, 1000)
    export(model, dummy)
    states = convert()

    assert sum(isinstance(m, EnBlock) for m in model.modules()) == 2
    assert sum(isinstance(m, DeBlock) for m in model.modules()) == 2
    assert len(states) == 6
    assert states.startup_samples == 14
    assert states.steady_samples == 4
    assert states.chunk_multiple == 4
    assert states.startup_remainder == 2
    for path in ("unet.onnx", "unet_streaming.onnx"):
        onnx.checker.check_model(onnx.load(path))

    stream_model = onnx.load("unet_streaming.onnx")
    pool_control_nodes = [
        node.name for node in stream_model.graph.node
        if "averagepool_window" in node.name
        or "averagepool_update_cache" in node.name
    ]
    assert not pool_control_nodes
    assert not any("averagepool_cache" in value.name
                   for value in [*stream_model.graph.input,
                                 *stream_model.graph.output])
    metadata = {item.key: item.value for item in stream_model.metadata_props}
    assert metadata["streaming.chunk_multiple"] == "4"
    assert metadata["streaming.startup_remainder"] == "2"

    session = ort.InferenceSession("unet.onnx", providers=["CPUExecutionProvider"])
    worst = 0.0
    for length in (255, 256, 1000, 1003, 1004):
        x = torch.randn(1, 1, length)
        with torch.no_grad():
            expected = model(x).numpy()
        actual = session.run(["output"], {"input": x.numpy()})[0]
        error = float(np.max(np.abs(expected - actual)))
        worst = max(worst, error)
        print(f"T={length}: output={actual.shape}, max_abs={error:.3e}")

    assert worst < 1e-5

    stream = ort.InferenceSession(
        "unet_streaming.onnx", providers=["CPUExecutionProvider"])
    x = torch.randn(1, 1, 1000).numpy()
    expected = session.run(["output"], {"input": x})[0]
    state_inputs = {}
    for state in states:
        state_inputs[state["cache_in"]] = np.zeros(
            (1, state["channels"], 0), np.float32)
    start = states.startup_samples
    results = stream.run(None, {"input": x[:, :, :start], **state_inputs})
    assert results[0].shape[2] == states.startup_outputs
    chunks = [results[0]]
    result_by_name = {
        output.name: value
        for output, value in zip(stream.get_outputs(), results)
    }
    state_inputs = {}
    for state in states:
        state_inputs[state["cache_in"]] = result_by_name[state["cache_out"]]
    block_sizes = (4, 8, 12, 20)
    aligned_end = start + (
        (x.shape[2] - start) // states.chunk_multiple
        * states.chunk_multiple)
    block_index = 0
    while start < aligned_end:
        block_size = block_sizes[block_index % len(block_sizes)]
        end = min(start + block_size, aligned_end)
        results = stream.run(None, {"input": x[:, :, start:end], **state_inputs})
        assert results[0].shape[2] >= states.steady_outputs
        chunks.append(results[0])
        result_by_name = {
            output.name: value
            for output, value in zip(stream.get_outputs(), results)
        }
        state_inputs = {}
        for state in states:
            state_inputs[state["cache_in"]] = result_by_name[state["cache_out"]]
        start = end
        block_index += 1
    assert start == aligned_end
    assert 0 <= x.shape[2] - start < states.chunk_multiple
    actual = np.concatenate(chunks, axis=2)
    assert actual.shape == expected.shape
    stream_error = float(np.max(np.abs(expected - actual)))
    assert stream_error < 1e-5

    print(f"全部通过，标准导出误差 {worst:.3e}，流式误差 {stream_error:.3e}，"
          f"启动 {states.startup_samples}，稳态最小输入 {states.steady_samples}")


if __name__ == "__main__":
    main()
