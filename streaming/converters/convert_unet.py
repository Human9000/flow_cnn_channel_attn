"""把 valid-convolution U-Net 转为逐张量状态的流式 ONNX。"""
import re

import numpy as np
import onnx
import onnx_graphsurgeon as gs
from onnx import helper, numpy_helper

INT64_MAX = np.iinfo(np.int64).max
TIME_AXIS = 2


class TensorShape:
    def __init__(self, dims):
        self.dims = tuple(int(dim) for dim in dims)


class ConversionResult(list):
    def __init__(self, states, schedule):
        super().__init__(states)
        self.startup_samples = schedule["startup_samples"]
        self.steady_samples = schedule["steady_samples"]
        self.startup_outputs = schedule["startup_outputs"]
        self.steady_outputs = schedule["steady_outputs"]
        self.chunk_multiple = schedule.get("chunk_multiple", 1)
        self.startup_remainder = schedule.get("startup_remainder", 0)


class InvalidTemporalLength(Exception):
    pass


def const(name, value):
    return gs.Constant(name, np.asarray(value, dtype=np.int64))


def clean(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


def node_attributes(node):
    return {attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute}


def temporal_output_length(model, input_length):
    values = {
        initializer.name: numpy_helper.to_array(initializer)
        for initializer in model.graph.initializer
    }
    model_input = model.graph.input[0]
    dims = []
    for index, dim in enumerate(model_input.type.tensor_type.shape.dim):
        if index == TIME_AXIS:
            dims.append(input_length)
        else:
            dims.append(dim.dim_value or 1)
    values[model_input.name] = TensorShape(dims)

    def array(name):
        return values[name]

    try:
        for node in model.graph.node:
            attrs = node_attributes(node)
            inputs = [values.get(name) for name in node.input]
            op = node.op_type
            if op == "Constant":
                result = numpy_helper.to_array(attrs["value"])
            elif op == "Identity":
                result = inputs[0]
            elif op in ("Relu", "BatchNormalization", "Softmax"):
                result = inputs[0]
            elif op == "Conv":
                source = inputs[0]
                weights = inputs[1]
                kernel = int(attrs.get("kernel_shape", weights.shape[2:])[-1])
                stride = int(attrs.get("strides", [1])[-1])
                dilation = int(attrs.get("dilations", [1])[-1])
                pads = attrs.get("pads", [0, 0])
                pad_left, pad_right = int(pads[-2]), int(pads[-1])
                effective = dilation * (kernel - 1) + 1
                time = ((source.dims[-1] + pad_left + pad_right - effective)
                        // stride + 1)
                if time <= 0:
                    raise InvalidTemporalLength
                result = TensorShape((source.dims[0], weights.shape[0], time))
            elif op == "AveragePool":
                source = inputs[0]
                kernel = int(attrs["kernel_shape"][-1])
                stride = int(attrs.get("strides", [1])[-1])
                pads = attrs.get("pads", [0, 0])
                numerator = source.dims[-1] + int(pads[-2]) + int(pads[-1]) - kernel
                if attrs.get("ceil_mode", 0):
                    time = (numerator + stride - 1) // stride + 1
                else:
                    time = numerator // stride + 1
                if time <= 0:
                    raise InvalidTemporalLength
                result = TensorShape((source.dims[0], source.dims[1], time))
            elif op == "Resize":
                source = inputs[0]
                if len(inputs) > 3 and inputs[3] is not None:
                    result = TensorShape(tuple(int(value) for value in inputs[3]))
                else:
                    scales = inputs[2]
                    result = TensorShape(
                        tuple(int(np.floor(dim * scale))
                              for dim, scale in zip(source.dims, scales)))
            elif op == "Shape":
                result = np.asarray(inputs[0].dims, dtype=np.int64)
            elif op == "Gather":
                result = np.take(inputs[0], inputs[1], axis=int(attrs.get("axis", 0)))
            elif op in ("Add", "Sub", "Mul", "Div"):
                left, right = inputs[:2]
                if isinstance(left, TensorShape):
                    if not isinstance(right, TensorShape) or left.dims != right.dims:
                        raise InvalidTemporalLength
                    result = left
                elif op == "Add":
                    result = left + right
                elif op == "Sub":
                    result = left - right
                elif op == "Mul":
                    result = left * right
                else:
                    result = np.trunc(left / right).astype(np.result_type(left, right))
            elif op == "Cast":
                dtype = helper.tensor_dtype_to_np_dtype(int(attrs["to"]))
                result = inputs[0].astype(dtype)
            elif op == "Unsqueeze":
                result = np.expand_dims(inputs[0], tuple(int(axis) for axis in inputs[1]))
            elif op == "Squeeze":
                axes = None if len(inputs) == 1 else tuple(int(axis) for axis in inputs[1])
                result = np.squeeze(inputs[0], axis=axes)
            elif op == "Concat":
                axis = int(attrs["axis"])
                if isinstance(inputs[0], TensorShape):
                    dims = list(inputs[0].dims)
                    dims[axis] = sum(value.dims[axis] for value in inputs)
                    result = TensorShape(dims)
                else:
                    result = np.concatenate(inputs, axis=axis)
            elif op == "Slice":
                source = inputs[0]
                starts = np.asarray(inputs[1]).reshape(-1)
                ends = np.asarray(inputs[2]).reshape(-1)
                axes = (np.arange(len(starts)) if len(inputs) < 4 or inputs[3] is None
                        else np.asarray(inputs[3]).reshape(-1))
                steps = (np.ones(len(starts), dtype=np.int64)
                         if len(inputs) < 5 or inputs[4] is None
                         else np.asarray(inputs[4]).reshape(-1))
                if isinstance(source, TensorShape):
                    dims = list(source.dims)
                    for start, end, axis, step in zip(starts, ends, axes, steps):
                        normalized = slice(int(start), int(end), int(step)).indices(
                            dims[int(axis)])
                        dims[int(axis)] = len(range(*normalized))
                    result = TensorShape(dims)
                else:
                    slices = [slice(None)] * source.ndim
                    for start, end, axis, step in zip(starts, ends, axes, steps):
                        slices[int(axis)] = slice(int(start), int(end), int(step))
                    result = source[tuple(slices)]
            else:
                raise ValueError(f"时间长度分析暂不支持 ONNX 算子: {op}")

            if len(node.output) != 1:
                raise ValueError(f"时间长度分析暂不支持多输出节点: {node.name}")
            values[node.output[0]] = result
    except InvalidTemporalLength:
        return 0

    output = array(model.graph.output[0].name)
    return output.dims[-1]


def analyze_streaming_schedule(src, search_limit=4096):
    model = onnx.load(src)
    startup = next(
        (length for length in range(1, search_limit + 1)
         if temporal_output_length(model, length) > 0), None)
    if startup is None:
        raise ValueError(f"无法在 {search_limit} 点内找到合法启动长度")

    probe_lengths = range(startup, startup + 64)
    steady = next(
        (step for step in range(1, 257)
         if all(temporal_output_length(model, length + step)
                > temporal_output_length(model, length)
                for length in probe_lengths)), None)
    if steady is None:
        raise ValueError("无法确定保证产生输出的稳态输入长度")

    startup_outputs = temporal_output_length(model, startup)
    steady_outputs = min(
        temporal_output_length(model, length + steady)
        - temporal_output_length(model, length)
        for length in probe_lengths)
    return {
        "startup_samples": startup,
        "steady_samples": steady,
        "startup_outputs": startup_outputs,
        "steady_outputs": steady_outputs,
    }


def infer_temporal_lengths(model, input_length):
    fixed = onnx.ModelProto()
    fixed.CopyFrom(model)
    time_dim = fixed.graph.input[0].type.tensor_type.shape.dim[TIME_AXIS]
    time_dim.ClearField("dim_param")
    time_dim.dim_value = input_length
    inferred = onnx.shape_inference.infer_shapes(fixed)

    lengths = {}
    values = [*inferred.graph.input, *inferred.graph.value_info,
              *inferred.graph.output]
    for value in values:
        dims = value.type.tensor_type.shape.dim
        if len(dims) > TIME_AXIS and dims[TIME_AXIS].HasField("dim_value"):
            lengths[value.name] = dims[TIME_AXIS].dim_value
    return lengths


def analyze_fixed_pool_phases(src, schedule):
    """提前证明相位对齐块下，每个 Pool 的尾部余数为常量。"""
    model = onnx.load(src)
    chunk_multiple = schedule["steady_samples"]
    boundaries = [
        schedule["startup_samples"] + index * chunk_multiple
        for index in range(8)
    ]
    lengths_by_boundary = [
        infer_temporal_lengths(model, length) for length in boundaries
    ]

    phases = {}
    for node in model.graph.node:
        if node.op_type != "AveragePool":
            continue
        stride = int(node_attributes(node)["strides"][-1])
        try:
            remainders = {
                lengths[node.input[0]] % stride
                for lengths in lengths_by_boundary
            }
        except KeyError as error:
            raise ValueError(
                f"无法提前确定 {node.name} 输入的时间长度") from error
        if len(remainders) != 1:
            raise ValueError(
                f"{node.name} 在 {chunk_multiple} 点对齐块下仍有动态相位: "
                f"{sorted(remainders)}")
        phases[node.name] = remainders.pop()
    return phases


def set_schedule_metadata(model, schedule):
    properties = {item.key: item.value for item in model.metadata_props}
    properties.update({
        "streaming.startup_samples": str(schedule["startup_samples"]),
        "streaming.steady_samples": str(schedule["steady_samples"]),
        "streaming.startup_outputs": str(schedule["startup_outputs"]),
        "streaming.steady_outputs": str(schedule["steady_outputs"]),
        "streaming.host_scheduled": "true",
    })
    if "chunk_multiple" in schedule:
        properties.update({
            "streaming.chunk_multiple": str(schedule["chunk_multiple"]),
            "streaming.startup_remainder": str(schedule["startup_remainder"]),
            "streaming.pool_phase_static": "true",
        })
    helper.set_model_props(model, properties)


class StateBuilder:
    def __init__(self, graph):
        self.graph = graph
        self.states = []

    def op(self, op, name, inputs, outputs, attrs=None):
        self.graph.nodes.append(gs.Node(
            op, name, attrs=attrs or {}, inputs=inputs, outputs=outputs))

    def length(self, tensor, prefix):
        shape = gs.Variable(f"{prefix}_shape", np.int64, [3])
        length = gs.Variable(f"{prefix}_length", np.int64, [1])
        self.op("Shape", f"{prefix}_get_shape", [tensor], [shape])
        self.op("Gather", f"{prefix}_get_length",
                [shape, const(f"{prefix}_time_axis", [TIME_AXIS])], [length],
                {"axis": 0})
        return length

    def slice(self, tensor, start, end, prefix):
        output = gs.Variable(f"{prefix}_slice", np.float32)
        self.op("Slice", f"{prefix}_slice_node",
                [tensor, start, end, const(f"{prefix}_axis", [TIME_AXIS])],
                [output])
        return output

    @staticmethod
    def rename_output(old, new):
        producer = old.inputs[0]
        producer.outputs = [new if value is old else value for value in producer.outputs]

    def add_tail(self, tensor, channels, prefix, count_mode, limit):
        cache_in = gs.Variable(
            f"{prefix}_cache_in", np.float32, [1, channels, f"{prefix}_cache_time"])
        cache_out = gs.Variable(
            f"{prefix}_cache_out", np.float32,
            [1, channels, f"{prefix}_cache_time_out"])
        window = gs.Variable(f"{prefix}_window", np.float32)
        self.op("Concat", f"{prefix}_concat", [cache_in, tensor], [window],
                {"axis": TIME_AXIS})

        if count_mode == "fixed":
            output_start = const(f"{prefix}_output_start", [-limit])
        else:
            window_length = self.length(window, f"{prefix}_window")
            remainder = gs.Variable(f"{prefix}_remainder", np.int64, [1])
            self.op("Mod", f"{prefix}_remainder_count",
                    [window_length, const(f"{prefix}_stride", [limit])], [remainder],
                    {"fmod": 0})
            output_start = gs.Variable(f"{prefix}_output_start", np.int64, [1])
            self.op("Sub", f"{prefix}_select_output_start",
                    [window_length, remainder], [output_start])
        tail = self.slice(window, output_start,
                          const(f"{prefix}_output_end", [INT64_MAX]),
                          f"{prefix}_update_cache")
        self.rename_output(tail, cache_out)

        self.graph.inputs.append(cache_in)
        self.graph.outputs.append(cache_out)
        self.states.append({
            "prefix": prefix, "cache_in": cache_in.name,
            "cache_out": cache_out.name, "channels": channels,
        })
        return window

    def add_skip_queue(self, skip, decoder, channels, prefix, initial_drop):
        cache_in = gs.Variable(
            f"{prefix}_cache_in", np.float32, [1, channels, f"{prefix}_cache_time"])
        cache_out = gs.Variable(
            f"{prefix}_cache_out", np.float32,
            [1, channels, f"{prefix}_cache_time_out"])

        cache_length = self.length(cache_in, f"{prefix}_cache_in")
        queue = gs.Variable(f"{prefix}_queue", np.float32)
        self.op("Concat", f"{prefix}_append", [cache_in, skip], [queue],
                {"axis": TIME_AXIS})

        first_chunk = gs.Variable(f"{prefix}_first_chunk", np.bool_, [1])
        first_chunk_int = gs.Variable(f"{prefix}_first_chunk_int", np.int64, [1])
        drop = gs.Variable(f"{prefix}_drop", np.int64, [1])
        self.op("Equal", f"{prefix}_is_first_chunk",
                [cache_length, const(f"{prefix}_zero", [0])], [first_chunk])
        self.op("Cast", f"{prefix}_first_chunk_to_int",
                [first_chunk], [first_chunk_int], {"to": 7})
        self.op("Mul", f"{prefix}_initial_drop",
                [first_chunk_int, const(f"{prefix}_drop_size", [initial_drop])], [drop])
        aligned = self.slice(queue, drop, const(f"{prefix}_aligned_end", [INT64_MAX]),
                             f"{prefix}_drop_left")
        decoder_length = self.length(decoder, f"{prefix}_decoder")
        paired = self.slice(aligned, const(f"{prefix}_paired_start", [0]),
                            decoder_length, f"{prefix}_paired")
        remaining = self.slice(aligned, decoder_length,
                               const(f"{prefix}_remaining_end", [INT64_MAX]),
                               f"{prefix}_remaining")
        self.rename_output(remaining, cache_out)
        self.graph.inputs.append(cache_in)
        self.graph.outputs.append(cache_out)
        self.states.append({
            "prefix": prefix, "cache_in": cache_in.name,
            "cache_out": cache_out.name, "channels": channels,
        })
        return paired


def convert(src="unet.onnx", dst="unet_streaming.onnx"):
    schedule = analyze_streaming_schedule(src)
    schedule["chunk_multiple"] = schedule["steady_samples"]
    schedule["startup_remainder"] = (
        schedule["startup_samples"] % schedule["chunk_multiple"])
    pool_phases = analyze_fixed_pool_phases(src, schedule)
    inferred = onnx.shape_inference.infer_shapes(onnx.load(src))
    graph = gs.import_onnx(inferred)
    graph.toposort()
    graph.inputs[0].shape = [1, 1, "chunk_time"]
    builder = StateBuilder(graph)

    for node in list(graph.nodes):
        if node.op == "Conv":
            weights = node.inputs[1].values
            kernel = weights.shape[-1]
            dilation = int(node.attrs.get("dilations", [1])[-1])
            pending = (kernel - 1) * dilation
            if pending:
                prefix = clean(node.name)
                node.inputs[0] = builder.add_tail(
                    node.inputs[0], weights.shape[1], prefix, "fixed", pending)
        elif node.op == "AveragePool":
            channels = int(node.inputs[0].shape[1])
            prefix = clean(node.name)
            remainder = pool_phases[node.name]
            if remainder:
                node.inputs[0] = builder.add_tail(
                    node.inputs[0], channels, prefix, "fixed", remainder)

    skip_specs = {
        "/de1/Slice": ("de1_skip", 32, 1),
        "/de2/Slice": ("de2_skip", 16, 6),
    }
    for node in list(graph.nodes):
        if node.name not in skip_specs:
            continue
        prefix, channels, initial_drop = skip_specs[node.name]
        add = node.outputs[0].outputs[0]
        decoder = add.inputs[0]
        paired = builder.add_skip_queue(
            node.inputs[0], decoder, channels, prefix, initial_drop)
        add.inputs = [decoder, paired]

    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    set_schedule_metadata(model, schedule)
    onnx.checker.check_model(model)
    onnx.save(model, dst)
    print(f"逐张量状态数量={len(builder.states)}，启动={schedule['startup_samples']}，"
          f"后续输入为 {schedule['chunk_multiple']} 的倍数")
    for state in builder.states:
        print(f"  {state['prefix']}: {state['cache_in']}")
    return ConversionResult(builder.states, schedule)


if __name__ == "__main__":
    convert()
