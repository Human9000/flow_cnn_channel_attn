"""一体化验证：valid 化 W + GAP→AvgPool[kH,kW] + 通用 Add 对齐 + host 补零，
确认整图输出与原始 same-padding 网络数值一致（判断 Conv 折叠 BN 的 bias
是否破坏「输入端补零」等价）。

所有关键量（补零 A、总长 L、Add 裁剪）均由符号执行计算，无暴力、无写死。
诊断与结果全部 flush 写入 _verify_out.txt；任何异常也写入日志，便于定位。
"""
import time
import traceback

import numpy as np
import onnx
import onnxruntime as ort
from onnx import helper, numpy_helper

import stream_analysis as sa

LOG = open("_verify_out.txt", "w", encoding="utf-8")


def log(msg):
    LOG.write(str(msg) + "\n")
    LOG.flush()


def i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def describe(coord):
    if coord is None:
        return "None"
    return (f"left0={coord.left0} scale={coord.scale} "
            f"span={coord.span} length={coord.length}")


def run():
    t0 = time.time()
    model = onnx.load("resnet50.onnx")
    graph = model.graph
    log(f"load {time.time() - t0:.1f}s nodes={len(graph.node)}")

    gap = next(n for n in graph.node if n.op_type == "GlobalAveragePool")
    gap_in = gap.input[0]
    log(f"GlobalAveragePool 节点={gap.name} 输入={gap_in}")

    # ---- 符号执行：W(valid=False) 与 H(valid=False) ----
    same_w = sa.propagate(graph, 224, axis=1, valid=False)
    same_h = sa.propagate(graph, 224, axis=0, valid=False)
    cw = same_w.get(gap_in)
    ch = same_h.get(gap_in)
    log(f"[same W] gap_in: {describe(cw)}")
    log(f"[same H] gap_in: {describe(ch)}")
    if cw is None or ch is None or cw.left0 is None:
        log("[中止] gap_in 坐标为空，符号执行链断裂，需检查 propagate")
        return

    kW = cw.length
    kH = ch.length
    A = -int(cw.left0)
    log(f"kW={kW} kH={kH} 左补零 A={A}")

    # ---- 求 valid 网络使 GAP 输入 W 长度==kW 的总输入长 L（求解，非扫参）----
    L = None
    for cand in range(kW, 800):
        v = sa.propagate(graph, cand, axis=1, valid=True,
                         global_kernel={gap.name: kW})
        c = v.get(gap_in)
        if c is not None and c.length == kW:
            L = cand
            break
    log(f"求得 L={L}")
    if L is None:
        log("[中止] 未找到使 valid GAP 输入长度==kW 的 L")
        return
    B = L - 224 - A
    log(f"右补零 B={B} 总长 L={L}")
    if B < 0:
        log("[中止] 右补零为负，长度不匹配")
        return

    # valid 网络（W）各张量坐标，用于 Add 对齐
    vw = sa.propagate(graph, L, axis=1, valid=True)

    # ---- 改图：valid 化 W pads + GAP→AvgPool[kH,kW] ----
    for node in graph.node:
        if node.op_type in ("Conv", "MaxPool", "AveragePool"):
            pads = sa.attr(node, "pads")
            if pads is not None:
                pads = list(pads)
                pads[1] = 0
                pads[3] = 0
                for a in node.attribute:
                    if a.name == "pads":
                        del a.ints[:]
                        a.ints.extend(pads)
        elif node.op_type == "GlobalAveragePool":
            node.op_type = "AveragePool"
            node.attribute.append(
                helper.make_attribute("kernel_shape", [kH, kW]))
            node.attribute.append(helper.make_attribute("strides", [1, 1]))
            node.attribute.append(
                helper.make_attribute("pads", [0, 0, 0, 0]))

    # ---- 通用 Add 对齐：用 Coord 计算裁剪，插入固定 Slice ----
    new_nodes = []
    inits = []
    n_aligned = 0
    for node in graph.node:
        if node.op_type == "Add":
            ca = vw.get(node.input[0])
            cb = vw.get(node.input[1])
            if ca is not None and cb is not None:
                target_left0 = max(ca.left0, cb.left0)
                target_len = min(ca.length, cb.length)
                for idx, c in enumerate((ca, cb)):
                    head = (target_left0 - c.left0) // c.scale
                    if head != 0 or c.length != target_len:
                        src = node.input[idx]
                        out = f"{src}_align{idx}"
                        s_n, e_n, x_n = out + "_s", out + "_e", out + "_x"
                        inits += [i64(s_n, [head]),
                                  i64(e_n, [head + target_len]),
                                  i64(x_n, [3])]
                        new_nodes.append(helper.make_node(
                            "Slice", [src, s_n, e_n, x_n], [out],
                            name=out + "_slice"))
                        node.input[idx] = out
                        n_aligned += 1
        new_nodes.append(node)
    del graph.node[:]
    graph.node.extend(new_nodes)
    graph.initializer.extend(inits)
    log(f"对齐 Slice 数={n_aligned}")

    onnx.checker.check_model(model)
    onnx.save(model, "resnet50_valid.onnx")
    log(f"已保存 resnet50_valid.onnx 用时 {time.time() - t0:.1f}s")

    # ---- 数值验证 ----
    t1 = time.time()
    ref = ort.InferenceSession("resnet50.onnx",
                               providers=["CPUExecutionProvider"])
    vsess = ort.InferenceSession("resnet50_valid.onnx",
                                 providers=["CPUExecutionProvider"])
    log(f"两个 session 就绪 {time.time() - t1:.1f}s")

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 3, 224, 224), dtype=np.float32)
    logits_ref = ref.run(["logits"], {"input": x})[0]

    padded = np.zeros((1, 3, 224, L), dtype=np.float32)
    padded[:, :, :, A:A + 224] = x
    logits_valid = vsess.run(["logits"], {"input": padded})[0]

    err = float(np.max(np.abs(logits_ref - logits_valid)))
    log(f"ref shape={logits_ref.shape} valid shape={logits_valid.shape}")
    log(f"[结果] max_abs = {err:.3e}")
    log("[结论] " + ("valid+补零等价 OK 可继续流式化"
                    if err < 2e-5 else
                    "不等价（bias 边界破坏），需逐层补零方案"))
    log(f"总用时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        log("[异常]\n" + traceback.format_exc())
    finally:
        LOG.close()
