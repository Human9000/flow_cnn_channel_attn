"""逐层补零验证（纯 onnx，无 onnx_graphsurgeon）。

每个 window 算子 valid 化 W + 前插 Pad(pl,pr on W) 复现 same padding。
每层补该层特征图真零 → 无 bias 污染，数学恒等于原始，预期 max_abs≈1e-6。
改图后做拓扑排序，保证节点顺序合法。结果写 _perlayer_out.txt。
"""
import re
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import stream_analysis as sa

LOG = open("_perlayer_out.txt", "w", encoding="utf-8")


def log(m):
    LOG.write(str(m) + "\n")
    LOG.flush()


def i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def clean(s):
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def toposort(nodes, available):
    """按数据依赖对节点做稳定拓扑排序。available: 已就绪张量名集合。"""
    produced = set(available)
    remaining = list(nodes)
    ordered = []
    while remaining:
        progressed = False
        for n in list(remaining):
            if all((inp == "" or inp in produced) for inp in n.input):
                ordered.append(n)
                produced.update(n.output)
                remaining.remove(n)
                progressed = True
        if not progressed:
            missing = {inp for n in remaining for inp in n.input
                       if inp and inp not in produced}
            raise RuntimeError(f"拓扑排序失败，缺失来源: {sorted(missing)[:5]}")
    return ordered


def run():
    t0 = time.time()
    model = onnx.load("resnet50.onnx")
    graph = model.graph
    log(f"load {time.time()-t0:.1f}s nodes={len(graph.node)}")

    gap = next(n for n in graph.node if n.op_type == "GlobalAveragePool")
    gap_in = gap.input[0]
    kW = sa.propagate(graph, 224, axis=1, valid=False)[gap_in].length
    kH = sa.propagate(graph, 224, axis=0, valid=False)[gap_in].length
    log(f"kW={kW} kH={kH}")

    new_nodes = []
    inits = []
    n_pad = 0
    for node in graph.node:
        if node.op_type in ("Conv", "MaxPool", "AveragePool"):
            pads = sa.attr(node, "pads")
            if pads is not None:
                pads = list(pads)
                pl, pr = int(pads[1]), int(pads[3])
                pads[1] = 0
                pads[3] = 0
                for a in node.attribute:
                    if a.name == "pads":
                        del a.ints[:]
                        a.ints.extend(pads)
                if pl or pr:
                    src = node.input[0]
                    tag = clean(node.name)
                    pout = f"pad_{tag}_out"
                    pname = f"pad_{tag}_pads"
                    inits.append(i64(pname, [0, 0, 0, pl, 0, 0, 0, pr]))
                    new_nodes.append(helper.make_node(
                        "Pad", [src, pname], [pout],
                        name=f"pad_{tag}", mode="constant"))
                    node.input[0] = pout
                    n_pad += 1
        elif node.op_type == "GlobalAveragePool":
            node.op_type = "AveragePool"
            node.attribute.append(helper.make_attribute("kernel_shape", [kH, kW]))
            node.attribute.append(helper.make_attribute("strides", [1, 1]))
            node.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))
        new_nodes.append(node)

    # 拓扑排序：起始可用张量 = 图输入 + 所有 initializer（含新增 pads）
    available = {v.name for v in graph.input}
    available |= {init.name for init in graph.initializer}
    available |= {init.name for init in inits}
    ordered = toposort(new_nodes, available)
    del graph.node[:]
    graph.node.extend(ordered)
    graph.initializer.extend(inits)
    log(f"插入 Pad 数={n_pad}，拓扑排序后节点={len(ordered)}")

    onnx.checker.check_model(model)
    onnx.save(model, "resnet50_perlayer.onnx")
    log(f"已保存 resnet50_perlayer.onnx 用时 {time.time()-t0:.1f}s")

    ref = ort.InferenceSession("resnet50.onnx",
                               providers=["CPUExecutionProvider"])
    psess = ort.InferenceSession("resnet50_perlayer.onnx",
                                 providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    logits_ref = ref.run(["logits"], {"input": x})[0]
    logits_per = psess.run(["logits"], {"input": x})[0]
    err = float(np.max(np.abs(logits_ref - logits_per)))
    log(f"ref={logits_ref.shape} perlayer={logits_per.shape}")
    log(f"[结果] 逐层Pad max_abs = {err:.3e}")
    log("[结论] " + ("逐层补零精确 OK，可流式化为 cache"
                    if err < 2e-5 else "仍有误差，需排查"))
    log(f"总用时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log("[异常]\n" + traceback.format_exc())
    finally:
        LOG.close()
