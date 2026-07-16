"""把当前一维 U-Net/旧 ResNetV2 ONNX 编译为 C99 actor runtime。"""
import argparse
import json
import re
from collections import deque
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

ALIGN_FLOATS = 4
EXTERNAL_NODE = 65535


def clean(name):
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_").lower()


class ActorGraph:
    def __init__(self, name, onnx_path):
        self.name = name
        self.model = onnx.load(onnx_path)
        self.node_by_name = {node.name: node for node in self.model.graph.node}
        self.constants = {
            item.name: numpy_helper.to_array(item).astype(np.float32)
            for item in self.model.graph.initializer
        }
        self.aliases = {
            node.output[0]: node.input[0]
            for node in self.model.graph.node if node.op_type == "Identity"
        }
        self.tensors = []
        self.tensor_by_name = {}
        self.nodes = []
        self.arrays = {}
        self.input = None
        self.output = None

    def resolve(self, name):
        while name in self.aliases:
            name = self.aliases[name]
        return self.constants[name]

    def tensor(self, name, channels):
        if name in self.tensor_by_name:
            return self.tensor_by_name[name]
        tensor_id = len(self.tensors)
        self.tensor_by_name[name] = tensor_id
        self.tensors.append({
            "id": tensor_id, "name": name, "channels": channels,
            "producer": -1, "consumers": [],
        })
        return tensor_id

    def add(self, name, op, inputs, output, output_channels, **params):
        node_id = len(self.nodes)
        output_id = self.tensor(output, output_channels)
        input_specs = []
        for input_name in inputs:
            tensor_id = self.tensor_by_name[input_name]
            reader = len(self.tensors[tensor_id]["consumers"])
            self.tensors[tensor_id]["consumers"].append(node_id)
            input_specs.append({"tensor": tensor_id, "reader": reader})
        self.tensors[output_id]["producer"] = node_id
        node = {
            "id": node_id, "name": name, "op": op,
            "inputs": input_specs, "output": output_id,
            "params": params,
        }
        self.nodes.append(node)
        return output_id

    def add_conv(self, name, onnx_node, input_name, output_name):
        node = self.node_by_name[onnx_node]
        attrs = {attr.name: helper.get_attribute_value(attr)
                 for attr in node.attribute}
        pads = tuple(int(value) for value in attrs.get("pads", ()))
        if any(pads):
            raise ValueError(
                f"C actor runtime 尚未实现 Conv padding: {onnx_node} "
                f"pads={pads}")
        weights = self.resolve(node.input[1])
        if weights.ndim != 3:
            raise ValueError(
                f"C actor runtime 只支持 Conv1d，{onnx_node} 权重为 "
                f"{weights.shape}")
        bias = self.resolve(node.input[2])
        weight_symbol = clean(f"{self.name}_{name}_weights")
        bias_symbol = clean(f"{self.name}_{name}_bias")
        self.arrays[weight_symbol] = weights
        self.arrays[bias_symbol] = bias
        return self.add(
            name, "CONV", [input_name], output_name, int(weights.shape[0]),
            in_channels=int(weights.shape[1]), out_channels=int(weights.shape[0]),
            kernel=int(weights.shape[2]), stride=1,
            weights=weight_symbol, bias=bias_symbol)

    def add_affine(self, name, onnx_node, input_name, output_name, channels):
        node = self.node_by_name[onnx_node]
        attrs = {attr.name: helper.get_attribute_value(attr)
                 for attr in node.attribute}
        gamma = self.resolve(node.input[1])
        beta = self.resolve(node.input[2])
        mean = self.resolve(node.input[3])
        variance = self.resolve(node.input[4])
        scale = gamma / np.sqrt(variance + float(attrs.get("epsilon", 1e-5)))
        bias = beta - mean * scale
        scale_symbol = clean(f"{self.name}_{name}_scale")
        bias_symbol = clean(f"{self.name}_{name}_bias")
        self.arrays[scale_symbol] = scale
        self.arrays[bias_symbol] = bias
        return self.add(
            name, "AFFINE", [input_name], output_name, channels,
            channels=channels, scale=scale_symbol, bias=bias_symbol)

    def finish(self, input_name, output_name):
        self.input = self.tensor_by_name[input_name]
        self.output = self.tensor_by_name[output_name]
        self.tensors[self.output]["consumers"].append(EXTERNAL_NODE)


def build_unet(path):
    graph = ActorGraph("unet_stream", path)
    graph.tensor("input", 1)
    graph.add_conv("en1_conv", "/en1/conv/Conv", "input", "en1_conv")
    graph.add("en1_relu", "RELU", ["en1_conv"], "en1_relu", 16, channels=16)
    graph.add("en1_pool", "AVGPOOL", ["en1_relu"], "en1_pool", 16,
              channels=16, kernel=2, stride=2)
    graph.add_conv("en2_conv", "/en2/conv/Conv", "en1_pool", "en2_conv")
    graph.add("en2_relu", "RELU", ["en2_conv"], "en2_relu", 32, channels=32)
    graph.add("en2_pool", "AVGPOOL", ["en2_relu"], "en2_pool", 32,
              channels=32, kernel=2, stride=2)
    graph.add("de1_up", "UPSAMPLE", ["en2_pool"], "de1_up", 32,
              channels=32, scale=2)
    graph.add_conv("de1_conv", "/de1/conv/Conv", "de1_up", "de1_conv")
    graph.add("de1_relu", "RELU", ["de1_conv"], "de1_relu", 32, channels=32)
    graph.add("de1_skip", "DROP", ["en2_relu"], "de1_skip", 32,
              channels=32, drop=1)
    graph.add("de1_add", "ADD", ["de1_relu", "de1_skip"], "de1_add", 32,
              channels=32)
    graph.add("de2_up", "UPSAMPLE", ["de1_add"], "de2_up", 32,
              channels=32, scale=2)
    graph.add_conv("de2_conv", "/de2/conv/Conv", "de2_up", "de2_conv")
    graph.add("de2_relu", "RELU", ["de2_conv"], "de2_relu", 16, channels=16)
    graph.add("de2_skip", "DROP", ["en1_relu"], "de2_skip", 16,
              channels=16, drop=6)
    graph.add("de2_add", "ADD", ["de2_relu", "de2_skip"], "de2_add", 16,
              channels=16)
    graph.add_conv("head", "/head/Conv", "de2_add", "logits")
    graph.add("softmax", "SOFTMAX", ["logits"], "output", 4, channels=4)
    graph.finish("input", "output")
    return graph


def build_resnet(path):
    graph = ActorGraph("resnetv2_stream", path)
    input_rank = len(graph.model.graph.input[0].type.tensor_type.shape.dim)
    if input_rank != 3:
        raise ValueError(
            f"build_resnet 仅支持旧 Conv1d 模型，当前输入 rank={input_rank}")
    graph.tensor("input", 1)
    graph.add_conv("stem", "/stem/Conv", "input", "stem")
    graph.add_affine("block1_bn", "/block1/bn1/BatchNormalization",
                     "stem", "block1_bn", 16)
    graph.add("block1_relu0", "RELU", ["block1_bn"], "block1_relu0", 16,
              channels=16)
    graph.add_conv("block1_conv1", "/block1/conv1/Conv",
                   "block1_relu0", "block1_conv1")
    graph.add("block1_relu1", "RELU", ["block1_conv1"], "block1_relu1", 16,
              channels=16)
    graph.add_conv("block1_conv2", "/block1/conv2/Conv",
                   "block1_relu1", "block1_main")
    graph.add("block1_skip", "DROP", ["stem"], "block1_skip", 16,
              channels=16, drop=2)
    graph.add("block1_add", "ADD", ["block1_main", "block1_skip"],
              "block1_add", 16, channels=16)
    graph.add_conv("block2_proj", "/block2/proj/Conv", "block1_add", "block2_proj")
    graph.add_affine("block2_bn", "/block2/bn1/BatchNormalization",
                     "block1_add", "block2_bn", 16)
    graph.add("block2_relu0", "RELU", ["block2_bn"], "block2_relu0", 16,
              channels=16)
    graph.add_conv("block2_conv1", "/block2/conv1/Conv",
                   "block2_relu0", "block2_conv1")
    graph.add("block2_relu1", "RELU", ["block2_conv1"], "block2_relu1", 32,
              channels=32)
    graph.add_conv("block2_conv2", "/block2/conv2/Conv",
                   "block2_relu1", "block2_main")
    graph.add("block2_skip", "DROP", ["block2_proj"], "block2_skip", 32,
              channels=32, drop=2)
    graph.add("block2_add", "ADD", ["block2_main", "block2_skip"],
              "block2_add", 32, channels=32)
    graph.add_affine("final_bn", "/bn/BatchNormalization",
                     "block2_add", "final_bn", 32)
    graph.add("final_relu", "RELU", ["final_bn"], "final_relu", 32,
              channels=32)
    graph.add_conv("head", "/head/Conv", "final_relu", "logits")
    graph.add("softmax", "SOFTMAX", ["logits"], "output", 4, channels=4)
    graph.finish("input", "output")
    return graph


def simulate(graph, samples=2048):
    writes = [0] * len(graph.tensors)
    reads = [[0] * len(tensor["consumers"]) for tensor in graph.tensors]
    maximum = [0] * len(graph.tensors)
    active_steps = [0] * len(graph.tensors)
    first_live = [None] * len(graph.tensors)
    last_live = [None] * len(graph.tensors)
    node_state = [node["params"].get("drop", 0) for node in graph.nodes]
    conflicts = [set() for _ in graph.tensors]
    queue = deque()
    queued = [False] * len(graph.nodes)
    event = 0
    post_push_live = [0] * len(graph.tensors)
    max_output_burst = 0

    def available(spec):
        return writes[spec["tensor"]] - reads[spec["tensor"]][spec["reader"]]

    def used(tensor_id):
        return writes[tensor_id] - min(reads[tensor_id])

    def snapshot():
        nonlocal event
        live = [index for index in range(len(graph.tensors)) if used(index) > 0]
        for tensor_id in live:
            count = used(tensor_id)
            maximum[tensor_id] = max(maximum[tensor_id], count)
            active_steps[tensor_id] += 1
            if first_live[tensor_id] is None:
                first_live[tensor_id] = event
            last_live[tensor_id] = event
        for index, left in enumerate(live):
            for right in live[index + 1:]:
                conflicts[left].add(right)
                conflicts[right].add(left)
        event += 1

    def ready(node_id):
        node = graph.nodes[node_id]
        first = available(node["inputs"][0])
        params = node["params"]
        if node["op"] == "CONV":
            return first >= params["kernel"]
        if node["op"] in ("RELU", "AFFINE", "UPSAMPLE", "SOFTMAX", "DROP"):
            return first >= 1
        if node["op"] == "AVGPOOL":
            return first >= params["kernel"]
        if node["op"] == "ADD":
            return first >= 1 and available(node["inputs"][1]) >= 1
        raise ValueError(node["op"])

    def enqueue(node_id):
        if node_id == EXTERNAL_NODE or queued[node_id] or not ready(node_id):
            return
        queued[node_id] = True
        queue.append(node_id)

    def notify(tensor_id):
        for consumer in graph.tensors[tensor_id]["consumers"]:
            enqueue(consumer)

    def execute(node_id):
        node = graph.nodes[node_id]
        params = node["params"]
        op = node["op"]
        input_ids = [item["tensor"] for item in node["inputs"]]
        touched = input_ids + [node["output"]]
        for left in touched:
            for right in touched:
                if left != right:
                    conflicts[left].add(right)
        if op == "CONV":
            count = ((available(node["inputs"][0]) - params["kernel"])
                     // params["stride"] + 1)
            consumed = count * params["stride"]
            produced = count
        elif op in ("RELU", "AFFINE", "SOFTMAX"):
            consumed = available(node["inputs"][0])
            produced = consumed
        elif op == "AVGPOOL":
            count = ((available(node["inputs"][0]) - params["kernel"])
                     // params["stride"] + 1)
            consumed = count * params["stride"]
            produced = count
        elif op == "UPSAMPLE":
            consumed = available(node["inputs"][0])
            produced = consumed * params["scale"]
        elif op == "ADD":
            consumed = min(available(node["inputs"][0]),
                           available(node["inputs"][1]))
            produced = consumed
            reads[node["inputs"][1]["tensor"]][node["inputs"][1]["reader"]] += consumed
        elif op == "DROP":
            available_count = available(node["inputs"][0])
            dropped = min(available_count, node_state[node_id])
            node_state[node_id] -= dropped
            consumed = available_count
            produced = available_count - dropped
        else:
            raise ValueError(op)
        reads[node["inputs"][0]["tensor"]][node["inputs"][0]["reader"]] += consumed
        writes[node["output"]] += produced
        snapshot()
        notify(node["output"])

    for _ in range(samples):
        writes[graph.input] += 1
        snapshot()
        notify(graph.input)
        while queue:
            node_id = queue.popleft()
            queued[node_id] = False
            if ready(node_id):
                execute(node_id)
        output_reader = len(graph.tensors[graph.output]["consumers"]) - 1
        output_burst = writes[graph.output] - reads[graph.output][output_reader]
        max_output_burst = max(max_output_burst, output_burst)
        reads[graph.output][output_reader] = writes[graph.output]
        snapshot()
        for tensor_id in range(len(graph.tensors)):
            if used(tensor_id) > 0:
                post_push_live[tensor_id] += 1

    capacities = [max(1, value) for value in maximum]
    return {
        "capacities": capacities,
        "conflicts": conflicts,
        "active_steps": active_steps,
        "first_live": first_live,
        "last_live": last_live,
        "persistent": [value > samples // 4 for value in post_push_live],
        "output_tokens": writes[graph.output],
        "max_output_burst": max_output_burst,
    }


def align(value, alignment=ALIGN_FLOATS):
    return (value + alignment - 1) // alignment * alignment


def allocate(graph, simulation):
    sizes = [simulation["capacities"][index] * tensor["channels"]
             for index, tensor in enumerate(graph.tensors)]
    order = sorted(range(len(sizes)),
                   key=lambda index: (-sizes[index],
                                      -len(simulation["conflicts"][index])))
    offsets = {}
    for tensor_id in order:
        candidates = {0}
        for other in simulation["conflicts"][tensor_id]:
            if other in offsets:
                candidates.add(align(offsets[other] + sizes[other]))
        for candidate in sorted(candidates):
            end = candidate + sizes[tensor_id]
            valid = True
            for other in simulation["conflicts"][tensor_id]:
                if other not in offsets:
                    continue
                other_start = offsets[other]
                other_end = other_start + sizes[other]
                if candidate < other_end and other_start < end:
                    valid = False
                    break
            if valid:
                offsets[tensor_id] = candidate
                break
        else:
            offsets[tensor_id] = align(
                max(offsets[index] + sizes[index] for index in offsets))
    arena_floats = align(max(offsets[index] + sizes[index] for index in offsets))
    return offsets, sizes, arena_floats


def float_literal(value):
    value = float(value)
    if not np.isfinite(value):
        raise ValueError("C 导出不支持非有限权重")
    text = f"{value:.9g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def render_array(name, values):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    lines = []
    for start in range(0, len(flat), 8):
        lines.append("    " + ", ".join(float_literal(value)
                                         for value in flat[start:start + 8]))
    return f"static const float {name}[{len(flat)}] = {{\n" + ",\n".join(lines) + "\n};\n"


def node_initializer(node, graph):
    inputs = ", ".join(
        f"{{{item['tensor']}, {item['reader']}}}" for item in node["inputs"])
    params = node["params"]
    op = node["op"]
    if op == "CONV":
        detail = (f".conv = {{{params['in_channels']}, {params['out_channels']}, "
                  f"{params['kernel']}, {params['stride']}, "
                  f"{params['weights']}, {params['bias']}}}")
    elif op == "AFFINE":
        detail = (f".affine = {{{params['channels']}, {params['scale']}, "
                  f"{params['bias']}}}")
    elif op == "AVGPOOL":
        detail = f".pool = {{{params['channels']}, {params['kernel']}, {params['stride']}}}"
    elif op == "UPSAMPLE":
        detail = f".upsample = {{{params['channels']}, {params['scale']}}}"
    else:
        detail = f".channel = {{{params['channels']}}}"
    initial_state = params.get("drop", 0)
    return (f"    {{.op=SR_OP_{op}, .input_count={len(node['inputs'])}, "
            f".inputs={{{inputs}}}, .output={node['output']}, "
            f".initial_state={initial_state}, .state={initial_state}, "
            f".params={{{detail}}}}}")


def emit(graph, output_dir):
    simulation = simulate(graph)
    offsets, sizes, arena_floats = allocate(graph, simulation)
    prefix = graph.name
    upper = prefix.upper()
    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_lines = []
    tensor_macros = []
    for tensor in graph.tensors:
        tensor_symbol = clean(tensor["name"]).upper()
        tensor_id = tensor["id"]
        consumers = [str(value) for value in tensor["consumers"]]
        consumers += ["0"] * (3 - len(consumers))
        tensor_lines.append(
            f"    {{.offset={offsets[tensor_id]}, .channels={tensor['channels']}, "
            f".capacity={simulation['capacities'][tensor_id]}, "
            f".producer={tensor['producer']}, .consumer_count={len(tensor['consumers'])}, "
            f".consumers={{{', '.join(consumers)}}}}}")
        tensor_macros.extend([
            f"#define {upper}_TENSOR_{tensor_symbol} {tensor_id}",
            f"#define {upper}_TENSOR_{tensor_symbol}_OFFSET_FLOATS {offsets[tensor_id]}",
            f"#define {upper}_TENSOR_{tensor_symbol}_OFFSET_BYTES {offsets[tensor_id] * 4}",
            f"#define {upper}_TENSOR_{tensor_symbol}_CAPACITY_TOKENS "
            f"{simulation['capacities'][tensor_id]}",
            f"#define {upper}_TENSOR_{tensor_symbol}_BYTES {sizes[tensor_id] * 4}",
        ])
    tensor_macro_text = "\n".join(tensor_macros)

    header = f'''#ifndef {upper}_H
#define {upper}_H

#include <stddef.h>
#include <stdint.h>
#include "stream_runtime.h"

#define {upper}_ARENA_FLOATS {arena_floats}
#define {upper}_TENSOR_COUNT {len(graph.tensors)}
#define {upper}_NODE_COUNT {len(graph.nodes)}
#define {upper}_OUTPUT_CHANNELS {graph.tensors[graph.output]['channels']}
#define {upper}_MAX_OUTPUT_TOKENS_PER_PUSH {simulation['max_output_burst']}

{tensor_macro_text}

typedef struct {{
    SR_ALIGN16 float arena[{upper}_ARENA_FLOATS];
    SrTensor tensors[{upper}_TENSOR_COUNT];
    SrNode nodes[{upper}_NODE_COUNT];
    uint16_t queue[{upper}_NODE_COUNT];
    SrRuntime runtime;
}} {''.join(part.capitalize() for part in prefix.split('_'))};

size_t {prefix}_context_size(void);
void {prefix}_init({''.join(part.capitalize() for part in prefix.split('_'))} *context);
void {prefix}_reset({''.join(part.capitalize() for part in prefix.split('_'))} *context);
int {prefix}_push(
    {''.join(part.capitalize() for part in prefix.split('_'))} *context,
    float input,
    float output[][4],
    uint32_t output_capacity_tokens);

#endif
'''
    context_type = "".join(part.capitalize() for part in prefix.split("_"))
    arrays = "\n".join(render_array(name, values)
                           for name, values in graph.arrays.items())
    tensor_initializers = ",\n".join(tensor_lines)
    node_initializers = ",\n".join(
        node_initializer(node, graph) for node in graph.nodes)
    source = f'''#include "{prefix}.h"

#include <string.h>

{arrays}
static const SrTensor tensor_template[{upper}_TENSOR_COUNT] = {{
{tensor_initializers}
}};

static const SrNode node_template[{upper}_NODE_COUNT] = {{
{node_initializers}
}};

size_t {prefix}_context_size(void) {{
    return sizeof({context_type});
}}

void {prefix}_init({context_type} *context) {{
    memset(context, 0, sizeof(*context));
    memcpy(context->tensors, tensor_template, sizeof(tensor_template));
    memcpy(context->nodes, node_template, sizeof(node_template));
    context->runtime.arena = context->arena;
    context->runtime.tensors = context->tensors;
    context->runtime.tensor_count = {upper}_TENSOR_COUNT;
    context->runtime.nodes = context->nodes;
    context->runtime.node_count = {upper}_NODE_COUNT;
    context->runtime.queue = context->queue;
    sr_runtime_reset(&context->runtime);
}}

void {prefix}_reset({context_type} *context) {{
    sr_runtime_reset(&context->runtime);
}}

int {prefix}_push(
    {context_type} *context,
    float input,
    float output[][4],
    uint32_t output_capacity_tokens) {{
    if (output_capacity_tokens < {upper}_MAX_OUTPUT_TOKENS_PER_PUSH) {{
        return -2;
    }}
    return sr_runtime_push(
        &context->runtime, {graph.input}, {graph.output}, &input,
        output_capacity_tokens ? &output[0][0] : NULL,
        output_capacity_tokens);
}}
'''
    (output_dir / f"{prefix}.h").write_text(header, encoding="utf-8")
    (output_dir / f"{prefix}.c").write_text(source, encoding="utf-8")

    plan = {
        "model": prefix,
        "layout": "time-major channel vectors",
        "allocation": "16-byte aligned conflict-aware static arena",
        "arena_bytes": arena_floats * 4,
        "parameter_bytes": sum(array.nbytes for array in graph.arrays.values()),
        "input_tensor": graph.tensors[graph.input]["name"],
        "output_tensor": graph.tensors[graph.output]["name"],
        "simulated_samples": 2048,
        "simulated_output_tokens": simulation["output_tokens"],
        "max_output_tokens_per_push": simulation["max_output_burst"],
        "tensors": [],
    }
    for tensor in graph.tensors:
        tensor_id = tensor["id"]
        shared = [other["name"] for other in graph.tensors
                  if other["id"] != tensor_id
                  and offsets[other["id"]] == offsets[tensor_id]]
        plan["tensors"].append({
            "id": tensor_id,
            "name": tensor["name"],
            "channels": tensor["channels"],
            "capacity_tokens": simulation["capacities"][tensor_id],
            "bytes": sizes[tensor_id] * 4,
            "offset_bytes": offsets[tensor_id] * 4,
            "producer": tensor["producer"],
            "consumers": tensor["consumers"],
            "first_live_event": simulation["first_live"][tensor_id],
            "last_live_event": simulation["last_live"][tensor_id],
            "active_events": simulation["active_steps"][tensor_id],
            "persistent": simulation["persistent"][tensor_id],
            "shares_offset_with": shared,
            "conflicts_with": [graph.tensors[other]["name"]
                               for other in sorted(simulation["conflicts"][tensor_id])],
        })
    (output_dir / f"{prefix}_memory_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("unet", "resnetv2", "all"), default="all")
    parser.add_argument("--output", default="c_generated")
    args = parser.parse_args()
    output_dir = Path(args.output)
    if args.model in ("unet", "all"):
        plan = emit(build_unet("unet.onnx"), output_dir)
        print(f"U-Net C arena: {plan['arena_bytes']} bytes")
    if args.model in ("resnetv2", "all"):
        plan = emit(build_resnet("resnetv2.onnx"), output_dir)
        print(f"ResNetV2 C arena: {plan['arena_bytes']} bytes")


if __name__ == "__main__":
    main()
