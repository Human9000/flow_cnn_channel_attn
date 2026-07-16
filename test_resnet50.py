"""timm 预训练 ResNet-50 的 PyTorch/ONNX 数值基线测试。"""
import numpy as np
import onnx
import onnxruntime as ort
import torch
from timm.models.resnet import Bottleneck

from convert_resnet50 import convert
from resnet50_model import (INPUT_SIZE, MODEL_NAME, NUM_CLASSES, ResNet50,
                            normalize_image)


ONNX_PATH = "resnet50.onnx"
STREAMING_ONNX_PATH = "resnet50_streaming.onnx"


def export(model, dummy, path=ONNX_PATH):
    torch.onnx.export(
        model,
        dummy,
        path,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch"},
        },
        opset_version=13,
        dynamo=False,
    )


def main():
    torch.manual_seed(0)
    model = ResNet50(pretrained=True).eval()
    normalized = normalize_image(torch.rand(1, *INPUT_SIZE))

    assert model.network.num_classes == NUM_CLASSES
    assert sum(isinstance(module, Bottleneck)
               for module in model.modules()) == 16
    assert model.network.conv1.padding == (3, 3)
    assert model.network.maxpool.padding == 1

    export(model, normalized)

    exported = onnx.load(ONNX_PATH)
    onnx.checker.check_model(exported)
    assert any(node.op_type == "GlobalAveragePool"
               for node in exported.graph.node)
    session = ort.InferenceSession(
        ONNX_PATH, providers=["CPUExecutionProvider"])

    worst = 0.0
    for height, width in ((1, 224), (1, 257)):
        sample = normalize_image(torch.rand(1, 3, height, width))
        with torch.no_grad():
            expected = model(sample).numpy()
        actual = session.run(["logits"], {"input": sample.numpy()})[0]
        error = float(np.max(np.abs(expected - actual)))
        worst = max(worst, error)
        assert actual.shape == (1, NUM_CLASSES)
        assert np.argmax(actual, axis=1).item() == np.argmax(
            expected, axis=1).item()
        assert error < 2e-5
        print(f"{height}x{width}: output={actual.shape}, max_abs={error:.3e}")
    print(f"{MODEL_NAME}: ONNX worst max_abs={worst:.3e}")

    convert(STREAMING_ONNX_PATH, width=224, pretrained=True)
    stream = ort.InferenceSession(
        STREAMING_ONNX_PATH, providers=["CPUExecutionProvider"])
    sample = normalize_image(torch.rand(1, *INPUT_SIZE)).numpy()
    expected = session.run(["logits"], {"input": sample})[0]
    cache = np.zeros((1, 3, 1, 0), np.float32)
    start = 0
    for block_size in (17, 31, 9, 53, 41, 73):
        end = min(start + block_size, sample.shape[3])
        eos = np.asarray([int(end == sample.shape[3])], np.int64)
        logits, cache = stream.run(None, {
            "rows": sample[:, :, :, start:end],
            "image_cache_in": cache,
            "eos": eos,
        })
        if eos.item():
            assert logits.shape == (1, NUM_CLASSES, 1)
            stream_error = float(np.max(np.abs(expected - logits[:, :, 0])))
            assert stream_error < 2e-5
            assert cache.shape[3] == 0
        else:
            assert logits.shape == (1, NUM_CLASSES, 0)
            assert cache.shape[3] == end
        start = end
        if start == sample.shape[3]:
            break
    assert start == sample.shape[3]
    print(f"按行变长块 + EOS: max_abs={stream_error:.3e}")


if __name__ == "__main__":
    main()
