import onnx
from onnx import numpy_helper

m = onnx.load("unet.onnx")
g = m.graph
consumers = {}
for n in g.node:
    for inp in n.input:
        consumers.setdefault(inp, []).append(n)
producer = {}
for n in g.node:
    for o in n.output:
        producer[o] = n
wshape = {i.name: list(numpy_helper.to_array(i).shape) for i in g.initializer}


def at(n, name):
    for a in n.attribute:
        if a.name == name:
            return list(a.ints) if a.ints else (a.i if a.type == 2 else None)
    return None


out = []
out.append("=== Conv ===")
for i, n in enumerate(g.node):
    if n.op_type == "Conv":
        out.append(f"#{i} {n.output[0][:36]:36s} k={at(n,'kernel_shape')} "
                   f"d={at(n,'dilations')} s={at(n,'strides')} w={wshape.get(n.input[1])}")
out.append("=== AveragePool/Resize ===")
for i, n in enumerate(g.node):
    if n.op_type in ("AveragePool", "Resize"):
        out.append(f"#{i} {n.op_type} k={at(n,'kernel_shape')} s={at(n,'strides')} out={n.output[0][:30]}")
out.append("=== Slice -> consumers ===")
for i, n in enumerate(g.node):
    if n.op_type == "Slice":
        cs = [c.op_type for c in consumers.get(n.output[0], [])]
        src = producer.get(n.input[0])
        out.append(f"#{i} from={src.op_type if src else '?'} out={n.output[0][:28]:28s} consumers={cs}")

open("analyze_out.txt", "w", encoding="utf-8").write("\n".join(out))
nodes = []
for i, n in enumerate(g.node):
    nodes.append(f"{i:3d} {n.op_type:13s} in={list(n.input)} -> {list(n.output)}")
open("nodes_full.txt", "w", encoding="utf-8").write("\n".join(nodes))
print("written", len(out), "summary lines and", len(nodes), "nodes")
