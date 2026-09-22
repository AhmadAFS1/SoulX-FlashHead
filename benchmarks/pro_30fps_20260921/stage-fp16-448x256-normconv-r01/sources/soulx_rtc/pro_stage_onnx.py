"""Graph rewrites for the exported Wan-decoder residual-stage ONNX before TensorRT builds it.

Why
---
Per-layer profiling of the shipped FP16 stage engines (RTX 4070 SUPER, TensorRT 10.9,
448x256 tail stage `upsamples[12-14]`, steady-state signature) put only 60% of the
engine's time in convolutions: 23% was **Reformat** layers, 6% Reduce, 7% pointwise, 4%
Slice. TensorRT runs the convolutions in a channel-vectorized layout (``1:8``, DHWC8) and
the RMS norm's channel reduction (``ReduceL2`` over C) plus the pointwise chain in the
linear layout, so it inserts a layout copy before and after every convolution. Forcing
DHWC8 network I/O does not help (the reduce still runs linear, 36 reformats instead of
29), and FP8 convolutions fall back to a slow generic "correlation" path on this
TensorRT/GPU pair, so the lever is the graph itself.

What
----
``clean`` (all exact, no numeric change):
  1. identity ``Cast`` fp16->fp16 removed (left over from the bf16->fp16 conversion);
  2. ``F.normalize``'s ``Shape``/``Expand`` broadcast removed (``Div`` broadcasts);
  3. ``Add(+0.0)`` removed;
  4. the dynamic ``Pad`` chain that ``torch.onnx.export`` emits for ``F.pad`` is folded into
     the convolution's own ``pads`` attribute (the temporal context comes from the cache
     concat, so only the spatial pads remain);
  5. dead nodes dropped.

``norm_conv`` (numerically equivalent, not bit-identical): the channel L2 norm
``sqrt(sum_c x_c^2)`` becomes a 1x1x1 convolution of ``(x/64)^2`` with all-ones weights,
followed by ``Sqrt`` and ``*64``. A convolution runs in the vectorized layout, so the norm,
the pointwise chain and the 3x3x3 convolutions share one layout. The 1/64 pre-scale keeps
the fp16 sum of squares below 65504 for activations up to |x| <= 2048 (observed max ~80)
and is exact (power of two). Measured on the tail stage: reformat layers 29 -> 20, Reduce
layers gone, 29.1 -> 26.1 ms per call (-10%); outputs differ from the shipped engine by
<= 0.16% of the tensor's max (fp16 accumulation order), with the shipped engine
bit-reproducible run to run.

Both are applied by ``benchmarks/pro_30fps_20260919/decoder_stages.py build
--graph-rewrite {none,clean,norm-conv}`` and recorded in the plan's results.
"""
from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _replace_nodes(graph, nodes) -> None:
    # Copy first: the repeated field owns its messages, and ``del graph.node[:]`` empties
    # any node object still referenced from ``nodes``.
    copies = []
    for node in nodes:
        copy = onnx.NodeProto()
        copy.CopyFrom(node)
        copies.append(copy)
    del graph.node[:]
    graph.node.extend(copies)


def _topological(graph, nodes):
    produced = {i.name for i in graph.input} | {i.name for i in graph.initializer}
    ordered, pending = [], list(nodes)
    while pending:
        progress = False
        for node in list(pending):
            if all(inp == "" or inp in produced for inp in node.input):
                ordered.append(node)
                produced.update(node.output)
                pending.remove(node)
                progress = True
        if not progress:
            raise RuntimeError("cycle in ONNX graph after rewrite")
    return ordered


def clean(model, *, norm_conv: bool = False) -> dict:
    """Rewrite ``model`` in place and return a small summary for provenance."""
    g = model.graph
    for index, node in enumerate(g.node):
        node.name = f"_n{index}_{node.op_type}"
    consts = {}
    for node in g.node:
        if node.op_type == "Constant":
            consts[node.output[0]] = numpy_helper.to_array(node.attribute[0].t)
    for init in g.initializer:
        consts[init.name] = numpy_helper.to_array(init)
    graph_outputs = {o.name for o in g.output}
    fp16_inputs = {i.name for i in g.input if i.type.tensor_type.elem_type == TensorProto.FLOAT16}
    producer = {o: node for node in g.node for o in node.output}
    removed: set[str] = set()
    before = {}
    for node in g.node:
        before[node.op_type] = before.get(node.op_type, 0) + 1

    def rewire(old, new):
        assert old not in graph_outputs, f"cannot drop the producer of graph output {old}"
        for node in g.node:
            for i, x in enumerate(node.input):
                if x == old:
                    node.input[i] = new

    # 1. identity casts
    activation_ops = ("Constant", "Shape", "Cast", "Reshape", "Transpose", "Slice", "ConstantOfShape")
    for node in list(g.node):
        if node.op_type != "Cast" or node.attribute[0].i != TensorProto.FLOAT16 or node.output[0] in graph_outputs:
            continue
        src = node.input[0]
        src_fp16 = src in fp16_inputs or (
            src in producer
            and (producer[src].op_type not in activation_ops
                 or (producer[src].op_type == "Concat" and any(a.name == "axis" and a.i == 2 for a in producer[src].attribute)))
        )
        if src_fp16:
            rewire(node.output[0], src)
            removed.add(node.name)
    # 2. Shape/Expand broadcast of the norm
    for node in list(g.node):
        if node.op_type == "Expand" and node.input[1] in producer and producer[node.input[1]].op_type == "Shape":
            rewire(node.output[0], node.input[0])
            removed.add(node.name)
    # 3. Add 0
    for node in list(g.node):
        if node.op_type == "Add" and node.output[0] not in graph_outputs:
            for k in (0, 1):
                c = consts.get(node.input[k])
                if c is not None and c.size == 1 and float(c) == 0.0:
                    rewire(node.output[0], node.input[1 - k])
                    removed.add(node.name)
                    break

    # 4. Pad -> Conv pads
    def eval_pads(name):
        node = producer[name]
        if node.op_type == "Cast":
            return eval_pads(node.input[0]).astype(np.int64)
        if node.op_type == "Reshape":
            return eval_pads(node.input[0]).reshape([int(s) for s in consts[node.input[1]]])
        if node.op_type == "Transpose":
            perm = [a.ints for a in node.attribute if a.name == "perm"][0]
            return np.transpose(eval_pads(node.input[0]), list(perm))
        if node.op_type == "Slice":
            data = eval_pads(node.input[0])
            starts, ends = consts[node.input[1]], consts[node.input[2]]
            axes = consts[node.input[3]] if len(node.input) > 3 and node.input[3] else np.arange(len(starts))
            steps = consts[node.input[4]] if len(node.input) > 4 and node.input[4] else np.ones(len(starts), dtype=np.int64)
            sl = [slice(None)] * data.ndim
            for s, e, a, st in zip(starts, ends, axes, steps):
                s, e, st = int(s), int(e), int(st)
                e = None if (st < 0 and e < -(2**62)) else e
                sl[int(a)] = slice(s, e, st)
            return data[tuple(sl)]
        if node.op_type == "Concat":
            axis = [a.i for a in node.attribute if a.name == "axis"][0]
            return np.concatenate([eval_pads(x) if x in producer else consts[x] for x in node.input], axis=axis)
        if node.op_type == "ConstantOfShape":
            value = numpy_helper.to_array(node.attribute[0].t) if node.attribute else np.zeros(1)
            return np.full([int(s) for s in consts[node.input[0]]], value.flatten()[0])
        if node.op_type == "Constant":
            return consts[name]
        raise ValueError(f"unexpected op in pad chain: {node.op_type}")

    folded_pads = []
    for node in list(g.node):
        if node.op_type != "Pad":
            continue
        pads = eval_pads(node.input[1]).astype(np.int64).flatten()
        rank = len(pads) // 2
        begin, end = pads[:rank], pads[rank:]
        if not (begin[0] == begin[1] == end[0] == end[1] == 0):
            raise ValueError(f"Pad touches batch/channel dims: {pads.tolist()}")
        consumers = [m for m in g.node if node.output[0] in m.input]
        if len(consumers) != 1 or consumers[0].op_type != "Conv":
            raise ValueError("Pad must feed exactly one Conv")
        conv = consumers[0]
        conv_pads = [int(v) for v in begin[2:]] + [int(v) for v in end[2:]]
        for attribute in conv.attribute:
            if attribute.name == "pads":
                if any(v != 0 for v in attribute.ints):
                    raise ValueError("Conv already carries padding")
                del attribute.ints[:]
                attribute.ints.extend(conv_pads)
        folded_pads.append(conv_pads)
        rewire(node.output[0], node.input[0])
        removed.add(node.name)

    # 4b. channel L2 norm as a 1x1x1 convolution
    norms = 0
    if norm_conv:
        new_nodes = []
        gamma_dims = {init.name: int(init.dims[0]) for init in g.initializer if init.name.endswith(".gamma")}

        def channels_of(norm_output):
            # Follow the norm's consumers (Max -> Div -> Mul(scale) -> Mul(gamma)) to the
            # block's gamma; stages mix channel counts (192 -> 384 in upsamples[4-6]).
            frontier, seen = [norm_output], set()
            while frontier:
                name = frontier.pop()
                for consumer in g.node:
                    if consumer.name in removed or name not in consumer.input:
                        continue
                    for inp in consumer.input:
                        if inp in gamma_dims:
                            return gamma_dims[inp]
                    for out in consumer.output:
                        if out not in seen:
                            seen.add(out)
                            frontier.append(out)
            raise ValueError(f"cannot find the gamma consuming {norm_output}")

        for node in list(g.node):
            if node.op_type != "ReduceL2" or node.name in removed:
                continue
            axes = consts.get(node.input[1])
            if axes is None or list(axes) != [1]:
                raise ValueError(f"ReduceL2 over unexpected axes {axes}")
            channels = channels_of(node.output[0])
            pre = f"norm{norms}_"
            norms += 1
            g.initializer.append(numpy_helper.from_array(np.ones((1, channels, 1, 1, 1), dtype=np.float16), pre + "ones"))
            new_nodes += [
                helper.make_node("Constant", [], [pre + "inv64"], name=pre + "c0",
                                 value=numpy_helper.from_array(np.array(1.0 / 64, dtype=np.float16), pre + "inv64_v")),
                helper.make_node("Constant", [], [pre + "x64"], name=pre + "c1",
                                 value=numpy_helper.from_array(np.array(64.0, dtype=np.float16), pre + "x64_v")),
                helper.make_node("Mul", [node.input[0], pre + "inv64"], [pre + "t"], name=pre + "scale"),
                helper.make_node("Mul", [pre + "t", pre + "t"], [pre + "sq"], name=pre + "square"),
                helper.make_node("Conv", [pre + "sq", pre + "ones"], [pre + "ss"], name=pre + "sum",
                                 kernel_shape=[1, 1, 1], pads=[0] * 6, strides=[1, 1, 1]),
                helper.make_node("Sqrt", [pre + "ss"], [pre + "n"], name=pre + "sqrt"),
                helper.make_node("Mul", [pre + "n", pre + "x64"], [node.output[0]], name=pre + "unscale"),
            ]
            removed.add(node.name)
        keep = [x for x in g.node if x.name not in removed] + new_nodes
        _replace_nodes(g, _topological(g, keep))
        removed = set()

    # 5. dead node elimination
    changed = True
    while changed:
        changed = False
        used = set()
        for node in g.node:
            if node.name not in removed:
                used.update(node.input)
        for node in g.node:
            if node.name in removed:
                continue
            if all(o not in used and o not in graph_outputs for o in node.output):
                removed.add(node.name)
                changed = True
    _replace_nodes(g, [node for node in g.node if node.name not in removed])
    after = {}
    for node in g.node:
        after[node.op_type] = after.get(node.op_type, 0) + 1
    return {"norm_conv": bool(norm_conv), "norms_rewritten": norms, "folded_conv_pads": folded_pads,
            "ops_before": before, "ops_after": after}
