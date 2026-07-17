"""生成并编译 C actor runtime，逐点对比标准 ONNX。"""
import ctypes
import json
import subprocess
from pathlib import Path

import numpy as np
import onnxruntime as ort

from streaming.converters.export_c import build_unet, emit

ROOT = Path(__file__).parent.parent
GENERATED = ROOT / "c_generated"


def compile_library():
    emit(build_unet(ROOT / "unet.onnx"), GENERATED)
    library = GENERATED / "streaming_models.dll"
    subprocess.run([
        "gcc", "-std=c99", "-O2", "-shared",
        f"-I{ROOT / 'c_runtime'}", f"-I{GENERATED}",
        str(ROOT / "c_runtime" / "stream_runtime.c"),
        str(GENERATED / "unet_stream.c"),
        "-o", str(library), "-lm",
    ], check=True)
    return library


def run_c(library, prefix, x, context=None):
    context_size = getattr(library, f"{prefix}_context_size")
    context_size.restype = ctypes.c_size_t
    initialize = getattr(library, f"{prefix}_init")
    initialize.argtypes = [ctypes.c_void_p]
    reset = getattr(library, f"{prefix}_reset")
    reset.argtypes = [ctypes.c_void_p]
    push = getattr(library, f"{prefix}_push")
    output_token = ctypes.c_float * 4
    push.argtypes = [ctypes.c_void_p, ctypes.c_float,
                     ctypes.POINTER(output_token), ctypes.c_uint32]
    push.restype = ctypes.c_int

    if context is None:
        context = ctypes.create_string_buffer(context_size())
        initialize(context)
    else:
        reset(context)
    output_buffer = (output_token * 8)()
    outputs = []
    first_output = None
    output_counts = set()
    for index, value in enumerate(x.reshape(-1)):
        count = push(context, float(value), output_buffer, 8)
        assert count >= 0
        output_counts.add(count)
        if count and first_output is None:
            first_output = index + 1
        for token_index in range(count):
            outputs.append(list(output_buffer[token_index]))
    return np.asarray(outputs, np.float32), first_output, output_counts, context


def validate_plan(path):
    plan = json.loads(path.read_text(encoding="utf-8"))
    for tensor in plan["tensors"]:
        assert tensor["offset_bytes"] % 16 == 0
        assert tensor["offset_bytes"] + tensor["bytes"] <= plan["arena_bytes"]
    by_name = {tensor["name"]: tensor for tensor in plan["tensors"]}
    for tensor in plan["tensors"]:
        start = tensor["offset_bytes"]
        end = start + tensor["bytes"]
        for other_name in tensor["conflicts_with"]:
            other = by_name[other_name]
            other_start = other["offset_bytes"]
            other_end = other_start + other["bytes"]
            assert end <= other_start or other_end <= start
    return plan


def main():
    library_path = compile_library()
    library = ctypes.CDLL(str(library_path.resolve()))
    x = np.random.default_rng(123).standard_normal((1, 1, 1000), dtype=np.float32)

    cases = (("unet_stream", "unet.onnx", 14, {0, 2, 4}),)
    for prefix, model_path, expected_first, expected_counts in cases:
        expected = ort.InferenceSession(
            str(ROOT / model_path), providers=["CPUExecutionProvider"]
        ).run(["output"], {"input": x})[0][0].T
        actual, first_output, output_counts, context = run_c(library, prefix, x)
        error = float(np.max(np.abs(expected - actual)))
        assert actual.shape == expected.shape
        assert first_output == expected_first
        assert output_counts == expected_counts
        assert error < 1e-5
        repeated, _, _, _ = run_c(library, prefix, x, context)
        assert np.array_equal(actual, repeated)
        plan = validate_plan(GENERATED / f"{prefix}_memory_plan.json")
        print(f"{prefix}: output={actual.shape}, max_abs={error:.3e}, "
              f"first={first_output}, arena={plan['arena_bytes']} bytes")

        long_x = np.random.default_rng(456).standard_normal(
            (1, 1, 4096), dtype=np.float32)
        long_expected = ort.InferenceSession(
            str(ROOT / model_path), providers=["CPUExecutionProvider"]
        ).run(["output"], {"input": long_x})[0][0].T
        long_actual, _, _, _ = run_c(library, prefix, long_x, context)
        long_error = float(np.max(np.abs(long_expected - long_actual)))
        assert long_actual.shape == long_expected.shape
        assert long_error < 1e-5


if __name__ == "__main__":
    main()
