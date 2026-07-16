"""把标准 resnet50.onnx 改造成逐张量状态缓存的流式 ONNX（纯 onnx，无 gs）。

方法（已验证逐层补零精确到 5.96e-7）：
  1. 时间轴 = W(dim3)；H 保留原 padding，W 方向 valid 化；
  2. GlobalAveragePool → AvgPool[kH,kW]（H 全局 + W 滑动，数值等价）；
  3. 每个 W 方向 kernel>1 的算子插入缓存：window=Concat(cache_in, x, rpad_in)，
     算子作用于 window，cache_out=window 尾部固定长度；
     cache_in 冷启动预填 pl 个零（左 pad），rpad_in 在 EOS 补 pr 个零（右 pad）；
  4. 残差 Add：短的主分支为基准，长的 shortcut 加 delay cache，用固定 Slice 对齐。

所有缓存长度由符号执行 + 含 pl 的数据流模拟静态算出，无动态 Shape/Gather/Mul。
"""
import re
import time
import traceback
from collections import defaultdict

import numpy as np
import onnx
from onnx import helper, numpy_helper

import stream_analysis as sa

W_AXIS = 3
INT64_MAX = np.iinfo(np.int64).max


def clean(s):
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_")


def i64(name, values):
    return numpy_helper.from_array(np.asarray(values, dtype=np.int64), name)


def wks(node):
    if node.op_type in ("Conv", "MaxPool", "AveragePool"):
        k = sa.attr(node, "kernel_shape")
        s = sa.attr(node, "strides", [1, 1])
        return int(k[-1]), int(s[-1])
    return None


def toposort(nodes, available):
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
            miss = {inp for n in remaining for inp in n.input
                    if inp and inp not in produced}
            raise RuntimeError(f"拓扑失败 缺:{sorted(miss)[:5]}")
    return ordered


def channel_height_maps(graph):
    """推每个张量的通道数 C 与 H 长度（H 保留 same padding，输入 224）。"""
    winit = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    cmap = {graph.input[0].name: 3}
    for node in graph.node:
        if node.op_type == "Conv":
            cmap[node.output[0]] = winit[node.input[1]].shape[0]
        elif node.op_type == "Gemm":
            cmap[node.output[0]] = winit[node.input[2]].shape[0]
        else:
            src = next((cmap[i] for i in node.input if i in cmap), None)
            if src is not None:
                cmap[node.output[0]] = src
    hmap = {name: (c.length if c else None)
            for name, c in sa.propagate(graph, 224, axis=0,
                                        valid=False).items()}
    return cmap, hmap


def simulate_limits(graph, gap, kW, pl_of, startup, steady, rounds=6):
    """含 pl 冷启动的数据流模拟：返回每个 window 节点稳态剩余(cache limit)。"""
    consumers = defaultdict(list)
    for ni, n in enumerate(graph.node):
        for ii, inp in enumerate(n.input):
            consumers[inp].append((ni, ii))
    in_name = graph.input[0].name

    def spec(n):
        if n is gap:
            return ("window", kW, 1)
        w = wks(n)
        if w:
            return ("window", w[0], w[1])
        if n.op_type == "Add":
            return ("add",)
        if n.op_type in ("Flatten", "Gemm", "Softmax"):
            return ("sink",)
        return ("pass",)

    buf = defaultdict(int)
    # 冷启动：每个 window 节点输入缓存预置 pl 个（左 pad 零，计数占位）
    for ni, n in enumerate(graph.node):
        if spec(n)[0] == "window":
            buf[(ni, 0)] = pl_of.get(n.name, 0)

    chunks = [startup] + [steady] * rounds
    rem_hist = []
    for chunk in chunks:
        for c in consumers[in_name]:
            buf[c] += chunk
        for ni, n in enumerate(graph.node):
            sp = spec(n)
            if sp[0] == "window":
                _, k, s = sp
                b = buf[(ni, 0)]
                fires = (b - k) // s + 1 if b >= k else 0
                buf[(ni, 0)] -= fires * s
                for c in consumers[n.output[0]]:
                    buf[c] += fires
            elif sp[0] == "pass":
                b = buf[(ni, 0)]
                buf[(ni, 0)] = 0
                for c in consumers[n.output[0]]:
                    buf[c] += b
            elif sp[0] == "add":
                a, bb = buf[(ni, 0)], buf[(ni, 1)]
                f = min(a, bb)
                buf[(ni, 0)] -= f
                buf[(ni, 1)] -= f
                for c in consumers[n.output[0]]:
                    buf[c] += f
            elif sp[0] == "sink":
                buf[(ni, 0)] = 0
        rem_hist.append({ni: buf[(ni, 0)]
                         for ni, n in enumerate(graph.node)
                         if spec(n)[0] == "window"})
    # 稳态取最后一轮
    return rem_hist[-1]


def convert(src="resnet50.onnx", dst="resnet50_streaming.onnx",
            log=print):
    t0 = time.time()
    model = onnx.load(src)
    graph = model.graph
    log(f"load {time.time()-t0:.1f}s nodes={len(graph.node)}")

    gap = next(n for n in graph.node if n.op_type == "GlobalAveragePool")
    gap_in = gap.input[0]
    sw = sa.propagate(graph, 224, axis=1, valid=False)
    sh = sa.propagate(graph, 224, axis=0, valid=False)
    kW = sw[gap_in].length
    kH = sh[gap_in].length
    startup = -sw[gap_in].left0 + 1  # 首次产出所需输入 = 感受野
    log(f"kW={kW} kH={kH} startup={startup}")

    cmap, hmap = channel_height_maps(graph)

    # 记录每个 window 算子 W 的 pl/pr（valid 化前）
    pl_of, pr_of, ks_of = {}, {}, {}
    for node in graph.node:
        w = wks(node)
        if w is not None:
            pads = sa.attr(node, "pads", [0, 0, 0, 0])
            pl_of[node.name] = int(pads[1]) if len(pads) == 4 else 0
            pr_of[node.name] = int(pads[3]) if len(pads) == 4 else 0
            ks_of[node.name] = w
    pl_of[gap.name] = 0
    pr_of[gap.name] = 0
    ks_of[gap.name] = (kW, 1)

    # 稳态 cache limit（含 pl 冷启动模拟）
    limits = simulate_limits(graph, gap, kW, pl_of, startup, 32)
    node_index = {id(n): i for i, n in enumerate(graph.node)}

    # valid 化 W + GAP→AvgPool[kH,kW]
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
            node.attribute.append(helper.make_attribute("kernel_shape", [kH, kW]))
            node.attribute.append(helper.make_attribute("strides", [1, 1]))
            node.attribute.append(helper.make_attribute("pads", [0, 0, 0, 0]))

    # valid 坐标（Add 对齐用）
    vw = sa.propagate(graph, startup + 32 * 4, axis=1, valid=True,
                      global_kernel={gap.name: kW})

    extra_nodes = []
    inits = []
    states = []
    new_inputs = []
    new_outputs = []

    def add_state(prefix, tensor, channels, height, kind, **extra):
        st = {"prefix": prefix, "channels": int(channels),
              "height": int(height), "kind": kind}
        st.update(extra)
        states.append(st)
        return st

    # 每个 window 算子（kW>1）插缓存
    for node in graph.node:
        if node.op_type not in ("Conv", "MaxPool", "AveragePool"):
            continue
        k, s = ks_of.get(node.name, (1, 1))
        if k <= 1:
            continue
        ni = node_index[id(node)]
        limit = limits[ni]
        pl = pl_of[node.name]
        pr = pr_of[node.name]
        src = node.input[0]
        C = cmap.get(src, 0)
        H = hmap.get(src, 0)
        tag = clean(node.name)
        cin = f"{tag}_cache_in"
        rin = f"{tag}_rpad_in"
        cout = f"{tag}_cache_out"
        window = f"{tag}_window"
        extra_nodes.append(helper.make_node(
            "Concat", [cin, src, rin], [window],
            name=f"{tag}_concat", axis=W_AXIS))
        node.input[0] = window
        # cache_out = window[-limit:]
        s_st, s_en, s_ax = f"{tag}_cs", f"{tag}_ce", f"{tag}_cx"
        inits += [i64(s_st, [-limit]), i64(s_en, [INT64_MAX]), i64(s_ax, [W_AXIS])]
        extra_nodes.append(helper.make_node(
            "Slice", [window, s_st, s_en, s_ax], [cout],
            name=f"{tag}_slice"))
        cin_vi = helper.make_tensor_value_info(
            cin, onnx.TensorProto.FLOAT, [1, C, H, None])
        rin_vi = helper.make_tensor_value_info(
            rin, onnx.TensorProto.FLOAT, [1, C, H, None])
        cout_vi = helper.make_tensor_value_info(
            cout, onnx.TensorProto.FLOAT, [1, C, H, None])
        new_inputs += [cin_vi, rin_vi]
        new_outputs.append(cout_vi)
        add_state(tag, src, C, H, "window",
                  cache_in=cin, rpad_in=rin, cache_out=cout,
                  pl=pl, pr=pr, limit=limit)

    # 残差 Add 对齐：short(input[1]) 加 delay cache，固定 Slice[:-diff]
    for node in graph.node:
        if node.op_type != "Add":
            continue
        ca = vw.get(node.input[0])
        cb = vw.get(node.input[1])
        if ca is None or cb is None:
            continue
        diff = cb.length - ca.length  # short 比 main 长的量
        if diff <= 0:
            continue
        short = node.input[1]
        C = cmap.get(short, 0)
        H = hmap.get(short, 0)
        tag = clean(node.name) + "_skip"
        cin = f"{tag}_cache_in"
        cout = f"{tag}_cache_out"
        queue = f"{tag}_queue"
        aligned = f"{tag}_aligned"
        extra_nodes.append(helper.make_node(
            "Concat", [cin, short], [queue], name=f"{tag}_concat", axis=W_AXIS))
        a_st, a_en, a_ax = f"{tag}_as", f"{tag}_ae", f"{tag}_ax"
        inits += [i64(a_st, [0]), i64(a_en, [-diff]), i64(a_ax, [W_AXIS])]
        extra_nodes.append(helper.make_node(
            "Slice", [queue, a_st, a_en, a_ax], [aligned], name=f"{tag}_slice_a"))
        c_st, c_en, c_ax = f"{tag}_cs", f"{tag}_ce", f"{tag}_cx"
        inits += [i64(c_st, [-diff]), i64(c_en, [INT64_MAX]), i64(c_ax, [W_AXIS])]
        extra_nodes.append(helper.make_node(
            "Slice", [queue, c_st, c_en, c_ax], [cout], name=f"{tag}_slice_c"))
        node.input[1] = aligned
        new_inputs.append(helper.make_tensor_value_info(
            cin, onnx.TensorProto.FLOAT, [1, C, H, None]))
        new_outputs.append(helper.make_tensor_value_info(
            cout, onnx.TensorProto.FLOAT, [1, C, H, None]))
        add_state(tag, short, C, H, "align", cache_in=cin, cache_out=cout,
                  limit=diff)

    all_nodes = list(graph.node) + extra_nodes
    available = {v.name for v in graph.input} | {i.name for i in graph.initializer}
    available |= {i.name for i in inits} | {vi.name for vi in new_inputs}
    ordered = toposort(all_nodes, available)
    del graph.node[:]
    graph.node.extend(ordered)
    graph.initializer.extend(inits)
    graph.input.extend(new_inputs)
    graph.output.extend(new_outputs)
    log(f"window/align 状态数={len(states)} 用时 {time.time()-t0:.1f}s")

    onnx.checker.check_model(model)
    props = {p.key: p.value for p in model.metadata_props}
    props.update({
        "streaming.startup_samples": str(startup),
        "streaming.steady_samples": "32",
        "streaming.time_axis": "3",
    })
    helper.set_model_props(model, props)
    onnx.save(model, dst)
    log(f"已保存 {dst} 用时 {time.time()-t0:.1f}s")
    return {"startup": startup, "steady": 32, "kW": kW, "kH": kH,
            "states": states}


if __name__ == "__main__":
    try:
        r = convert()
        with open("_convert_out.txt", "w", encoding="utf-8") as fh:
            fh.write(f"startup={r['startup']} steady={r['steady']} "
                     f"kW={r['kW']} kH={r['kH']} states={len(r['states'])}\n")
            for st in r["states"]:
                fh.write(str(st) + "\n")
    except Exception:
        with open("_convert_out.txt", "w", encoding="utf-8") as fh:
            fh.write(traceback.format_exc())
