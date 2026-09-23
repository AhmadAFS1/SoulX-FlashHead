"""DiT transformer-block skipping for the Wan 1.3B FlashHead denoiser (flag-gated).

WHY THIS EXISTS
---------------
At 576x320 the DiT forward is ~394 ms of a 1,427 ms window (two distilled sampling
steps over 30 ``DiTAudioBlock``s, so ~6.6 ms per block per step). It is the second
largest term after the VAE decoder, and the only one whose cost is a plain count of
identical units. Removing k of the 30 blocks removes k/30 of the DiT time with no
kernel work and no re-export; whether the video survives it is a question for the
quality gate (the user accepts the risk if it is validated on video), not for this
module. This module's job is a *correct, switchable* implementation with a one-flag
rollback (``run.py --skip-dit-blocks`` absent == stock model).

TWO MODES
---------
``install_dit_block_skip(model, indices, step=None)``:

* ``step is None`` -- **drop**: ``model.blocks`` becomes an ``nn.ModuleList`` of the
  survivors, in their original order. Every consumer in the repository iterates
  ``model.blocks`` (see "INDEX SAFETY" below), so the surviving blocks are simply
  renumbered ``0..len-1`` and nothing needs to know the original indices. This is
  the cheapest variant: the dropped blocks never run at any step and Dynamo never
  even sees them.

* ``step is an int`` -- **step-aware bypass**: the blocks stay in place but the listed
  ones return the residual stream unchanged when ``CURRENT_STEP[0] == step``.
  ``CURRENT_STEP[0]`` is the 0-based sampling-step index that ``run.py``'s
  ``prepared_forward`` writes before every DiT call (the distilled schedule is
  ``[1000, 625, 0]``: step 0 is the high-noise structure pass, step 1 the refinement
  pass). This lets the coordinator test "skip only during refinement" without
  touching the schedule.

WHAT DYNAMO DOES WITH THE STEP-AWARE WRAPPER
--------------------------------------------
``run.py`` compiles ``model.forward`` with ``torch.compile(dynamic=False,
fullgraph=False)``. The wrapper is an instance attribute ``block.forward`` (a closure
over ``step`` and the original bound ``DiTAudioBlock.forward``); Dynamo inlines it like
any Python function reached through ``nn.Module.__call__``. The read of the
module-global ``CURRENT_STEP[0]`` is a Python ``int``, so Dynamo *constant-folds* the
``if`` and installs a guard on the value of ``CURRENT_STEP[0]``. Consequences:

* the compiled graph for step 0 and the compiled graph for step 1 are two separate
  specialisations (two Inductor compiles at warm-up instead of one, well inside the
  default ``cache_size_limit`` of 8); after warm-up each DiT call hits its cached
  graph with a cheap guard check;
* in the specialisation where the block is bypassed, the block's kernels are simply
  absent from the graph -- no branch, no ``torch.where``, no wasted compute;
* nothing in the wrapper graph-breaks: no prints, no ``.item()``, no data-dependent
  Python on tensors. ``fullgraph=False`` is not relied upon.

WHEN THE STEP-AWARE MODE CAN SILENTLY DO NOTHING
-------------------------------------------------
``CURRENT_STEP`` is written only by ``run.py``'s ``prepared_forward`` (policy key
``prepared_conditioning: true``), and a step index that the schedule never reaches
(``--dit-skip-step 2`` with ``--sampling-steps 2``) is never observed. In both cases a
``step=k`` install succeeds and bypasses nothing: the run would measure the stock
model under a "skip" label. This module cannot see the policy or the schedule at
install time, so it does two things instead: the manifest carries a ``requires``
string that names both preconditions (the coordinator's gate must check
``results["prepared_conditioning"]["enabled"]`` and
``results["sampling_schedule"]["sampling_steps"] > step``), and an optional
``sampling_steps=`` keyword lets a caller that knows the schedule fail at install
time. No hit counter is kept inside the wrapper on purpose: mutating a global under
Dynamo would be a read-then-write of guarded state and would recompile every call.

INT8 CALIBRATION AND DROP MODE DO NOT MIX
-----------------------------------------
``run.py --calibrate-int8`` arms ``Int8ComputeLinear.calibrating`` BEFORE the skip is
installed and reads the observed amax back AFTER generation with
``collect_int8_amax(model)``, which labels every site by ``model.named_modules()``
name. After a drop the survivors are renumbered, so original block 8's FFN would be
written as ``blocks.7.ffn.0`` and the last k sites would vanish; if that JSON were
later fed to ``--static-int8-scales`` (applied on stock names before the skip) the
wrong clip range would land on the wrong block with no error anywhere. Drop mode
therefore refuses to install while any module is calibrating; calibrate without
``--skip-dit-blocks`` or with ``--dit-skip-step`` (names preserved). The drop
manifest also carries ``original_names`` (``blocks.<new>`` -> ``blocks.<original>``)
for any consumer that must map renumbered names back.

INDEX SAFETY (what was checked, 2026-09-22)
-------------------------------------------
Dropping blocks renumbers the survivors. This is safe because ``run.py`` installs the
skip AFTER everything that is keyed by the original names (``blocks.17.ffn.2`` etc.)
and BEFORE everything that iterates the surviving list:

* ``flash_head/src/modules/flash_head_model.py``: ``prepare_conditioning`` builds
  ``cross_kv`` as ``tuple(block.cross_attn.prepare_kv(flat) for block in self.blocks)``
  and ``forward_blocks`` does ``for index, block in enumerate(self.blocks): ...
  cross_kv[index]`` -- both over the *current* list, so the tuple and the loop stay
  aligned after a drop. ``DiTAudioBlock.i`` / ``num_layers`` are stored in ``__init__``
  and read nowhere (``grep -rn "\\.i\\b\\|num_layers"`` over ``flash_head/src``,
  ``soulx_rtc``, ``benchmarks/pro_quantization_v2_20260918``: only the assignments).
* ``soulx_rtc/pro_quantization_v2.py`` (policy resolution, FP8/INT8 conversion),
  ``soulx_rtc/pro_quantization.py`` (static INT8 scales, ``blocks.{i}.ffn.{j}``),
  ``soulx_rtc/pro_attention_backends.py`` (``install_self_attention_backend`` sets
  ``block.self_attn._pro_attention_backend`` per module object; ``remove_*`` iterates
  the list) -- all run before the skip and attach state to the module objects, which
  survive renumbering untouched.
* ``benchmarks/pro_quantization_v2_20260918/run.py``: after the skip only
  ``_install_prepared_generation`` (iterates nothing by index), the ``ffn_only``
  compile loop (``for block in pipeline.model.blocks``) and the optional capture /
  profile / latency attachments run. ``capture.py`` (``--capture-manifest`` only)
  enumerates the surviving list and hard-codes representative paths
  ``blocks.{0,14,29}.self_attn.*`` for ``get_submodule``; with a drop that shortens
  the list below 30 a capture run would fail on ``blocks.29`` -- capture is a
  diagnostic mode that is never combined with a skip experiment, and the failure is
  loud, not silent.
* ``flash_head/src/pipeline/flash_head_pipeline.py`` (``generate``) never touches
  ``model.blocks``. ``soulx_rtc/trt_backend.py`` / ``trt_experiment.py`` index
  ``model.blocks[index]`` but belong to the retired v1 TensorRT-FFN path, not the v2
  runner.
* The diffusers ``model.config.num_layers`` still says 30 after a drop; it is only
  read by ``__init__``.

``remove_dit_block_skip(model)`` restores the original list / forwards. It must run
BEFORE ``_install_prepared_generation`` re-wraps ``model.forward`` (a compiled graph
captured with the skip in place would otherwise stay in Dynamo's cache). In
step-bypass mode it only unwinds its own closure: if something else has since been
layered on ``block.forward`` (``flash_head/utils/latency.py`` ``wrap(module,
"forward")`` in the eager-diagnostic path does exactly that) it raises instead of
deleting the outer wrapper, mirroring the install-time "refusing to stack" guard.
"""
from __future__ import annotations

from typing import Any, Sequence

import torch.nn as nn

# 0-based sampling-step index of the DiT call in flight. run.py's prepared_forward
# assigns CURRENT_STEP[0] = index before every base_forward call (a list, not a bare
# int, so the assignment is visible through the module import without `global`).
CURRENT_STEP = [0]

# The Wan 1.3B FlashHead DiT has exactly 30 blocks; a different count means the
# caller handed us the wrong module (or a future checkpoint) and every index set in
# suggest_skip_sets() would be wrong for it.
EXPECTED_BLOCKS = 30

_STATE_ATTR = "_pro_dit_block_skip"


def _validate(model: nn.Module, indices: Sequence[int], expected_blocks: int) -> list[int]:
    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, nn.ModuleList):
        raise TypeError("model.blocks must be an nn.ModuleList of transformer blocks")
    if getattr(model, _STATE_ATTR, None) is not None:
        raise RuntimeError("DiT block skip already installed; call remove_dit_block_skip first")
    if len(blocks) != expected_blocks:
        raise AssertionError(
            f"expected {expected_blocks} DiT blocks (stock Wan 1.3B FlashHead), found {len(blocks)}"
        )
    skipped: list[int] = []
    for raw in indices:
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"block index {raw!r} is not an int")
        if not 0 <= raw < len(blocks):
            raise IndexError(f"block index {raw} outside 0..{len(blocks) - 1}")
        if raw not in skipped:
            skipped.append(raw)
    if not skipped:
        raise ValueError("no block indices given; omit the flag instead of passing an empty list")
    if len(skipped) == len(blocks):
        raise ValueError("refusing to skip every block")
    return sorted(skipped)


def _make_step_bypass(block: nn.Module, step: int, original_forward):
    """Closure that returns the residual stream unchanged at sampling step ``step``.

    ``DiTAudioBlock.forward(x, context, t_mod, freqs, grid_sizes, cross_kv=None,
    rotary=None)`` returns the single tensor ``x`` (residual stream, shape
    ``(b, f*h*w, dim)``); returning the input ``x`` object is exactly what the block
    computes with all three residual branches zeroed, and ``forward_blocks`` only
    re-assigns it, so aliasing is harmless. Positional/keyword pass-through keeps the
    wrapper agnostic to the exact call site (``forward_blocks`` passes ``cross_kv``
    and ``rotary`` positionally).
    """

    def bypass_forward(x, *args, **kwargs):
        # Plain Python int compare -> constant-folded by Dynamo with a guard on
        # CURRENT_STEP[0]; two step values give two graph specialisations.
        if CURRENT_STEP[0] == step:
            return x
        return original_forward(x, *args, **kwargs)

    bypass_forward._pro_dit_skip_step = step  # type: ignore[attr-defined]
    bypass_forward._pro_dit_original_forward = original_forward  # type: ignore[attr-defined]
    return bypass_forward


def install_dit_block_skip(
    model: nn.Module,
    indices: Sequence[int],
    step: int | None = None,
    *,
    expected_blocks: int = EXPECTED_BLOCKS,
    offload_dropped: bool = False,
    sampling_steps: int | None = None,
) -> dict[str, Any]:
    """Skip DiT blocks ``indices`` (0-based, original numbering) and return a manifest.

    ``step=None`` drops the blocks from ``model.blocks`` (survivors renumbered; see the
    module docstring for why that is safe at run.py's install point). ``step=k``
    keeps the list intact and bypasses the listed blocks only while
    ``CURRENT_STEP[0] == k``.

    ``offload_dropped=True`` moves the dropped blocks to the CPU so their weights
    (~87 MB each in bf16 at 1.3B) leave the GPU; the default keeps them where they are
    so ``remove_dit_block_skip`` is an exact, instantaneous rollback. Only meaningful
    in drop mode.

    ``sampling_steps`` (optional) is the number of DiT passes per window; when given,
    a ``step`` the schedule never reaches is rejected here instead of producing a
    run that measures the stock model (see the module docstring). run.py does not
    pass it today, so the manifest's ``requires`` field is the audit trail.

    The returned manifest is JSON-serialisable and is what run.py stores under
    ``result["dit_ops"]`` so a results file says which blocks a video was made without.
    """
    skipped = _validate(model, indices, expected_blocks)
    total = len(model.blocks)
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError(f"step must be None or a non-negative int, got {step!r}")
    if step is not None and sampling_steps is not None and step >= sampling_steps:
        raise ValueError(
            f"dit skip step {step} is never reached with {sampling_steps} sampling steps "
            f"(valid: 0..{sampling_steps - 1}); the bypass would silently never fire"
        )

    if step is None:
        # Renumbering survivors would corrupt an INT8 calibration run: the amax is
        # collected after generation by named_modules() name (see module docstring).
        armed = [name for name, m in model.named_modules() if getattr(m, "calibrating", False)]
        if armed:
            raise RuntimeError(
                f"drop mode renumbers blocks but {len(armed)} INT8 site(s) are calibrating "
                f"(first: {armed[0]}); collect_int8_amax would mislabel them. Calibrate with "
                "--dit-skip-step (names preserved) or without --skip-dit-blocks."
            )
        original = model.blocks
        survivors = [block for index, block in enumerate(original) if index not in skipped]
        dropped = [original[index] for index in skipped]
        if offload_dropped:
            for block in dropped:
                block.to("cpu")
        # nn.Module.__setattr__ replaces the registered child; parameters of the
        # dropped blocks disappear from model.parameters()/state_dict() until removed.
        model.blocks = nn.ModuleList(survivors)
        state = {"mode": "drop", "original_blocks": original, "dropped": dropped, "offloaded": offload_dropped}
        surviving = [i for i in range(total) if i not in skipped]
        manifest: dict[str, Any] = {
            "mode": "drop",
            "skipped": skipped,
            "kept": len(survivors),
            "surviving_original_indices": surviving,
            # named_modules() names after the drop -> stock names, for anything that
            # records per-module state by name after this point (only the entries
            # that actually changed; blocks before the first skipped index keep theirs).
            "original_names": {
                f"blocks.{new}": f"blocks.{orig}" for new, orig in enumerate(surviving) if new != orig
            },
            "step": None,
            "original_block_count": total,
            "offload_dropped": offload_dropped,
            "expected_dit_time_fraction_removed": len(skipped) / total,
        }
    else:
        wrapped: list[tuple[nn.Module, Any]] = []
        for index in skipped:
            block = model.blocks[index]
            # Only ever wrap the class forward; a pre-existing instance attribute
            # would mean someone else already replaced it and we must not stack.
            if "forward" in block.__dict__:
                raise RuntimeError(f"blocks.{index}.forward is already an instance attribute; refusing to stack")
            closure = _make_step_bypass(block, step, block.forward)
            block.forward = closure
            wrapped.append((block, closure))
        state = {"mode": "step_bypass", "wrapped": wrapped, "step": step}
        manifest = {
            "mode": "step_bypass",
            "skipped": skipped,
            "kept": total,  # module count is unchanged; the bypass is dynamic
            "active_blocks_at_step": {str(step): total - len(skipped)},
            "step": step,
            "original_block_count": total,
            "dynamo": "CURRENT_STEP[0] is guarded; expect one extra DiT specialisation",
            # Preconditions this module cannot verify itself (see module docstring);
            # the coordinator's gate checks them against the results file.
            "requires": f"prepared_conditioning=true and sampling_steps > {step}",
            "sampling_steps_checked": sampling_steps,
            # Two sampling steps in the shipping schedule: bypassing at one of them
            # removes half of what a drop would.
            "expected_dit_time_fraction_removed": len(skipped) / total / 2,
        }
    setattr(model, _STATE_ATTR, state)
    return manifest


def remove_dit_block_skip(model: nn.Module) -> dict[str, Any]:
    """Undo ``install_dit_block_skip``; returns ``{"restored": mode}`` or ``{}`` if none."""
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return {}
    if state["mode"] == "drop":
        original: nn.ModuleList = state["original_blocks"]
        if state["offloaded"]:
            device = next(model.blocks.parameters()).device
            for block in state["dropped"]:
                block.to(device)
        model.blocks = original
    else:
        # Unwind only our own layer: if something wrapped block.forward after the
        # install (latency instrumentation, a profiler), deleting the instance
        # attribute would silently discard that outer wrapper too.
        for block, closure in state["wrapped"]:
            current = block.__dict__.get("forward")
            if current is not closure:
                # functools.wraps copies our __qualname__ onto an outer wrapper, so
                # name it by identity and by the __wrapped__ marker, not by name.
                raise RuntimeError(
                    "block.forward is no longer the step-bypass closure installed here: "
                    f"found a different object at 0x{id(current):x} "
                    f"(functools.wraps layer: {hasattr(current, '__wrapped__')}); "
                    "remove the outer wrapper first, then remove_dit_block_skip"
                )
        for block, _ in state["wrapped"]:
            del block.__dict__["forward"]  # falls back to the class method again
    delattr(model, _STATE_ATTR)
    return {"restored": state["mode"]}


def describe_dit_block_skip(model: nn.Module) -> dict[str, Any]:
    """Read-only view of an installed skip (for results files and assertions)."""
    state = getattr(model, _STATE_ATTR, None)
    if state is None:
        return {"installed": False}
    if state["mode"] == "drop":
        return {"installed": True, "mode": "drop", "kept": len(model.blocks),
                "dropped": len(state["dropped"]), "offloaded": state["offloaded"]}
    return {"installed": True, "mode": "step_bypass", "step": state["step"],
            "wrapped": len(state["wrapped"]), "kept": len(model.blocks)}


def _evenly_spaced_middle(k: int, total: int = EXPECTED_BLOCKS) -> list[int]:
    """k indices spread evenly over the open interval (0, total-1), never the ends."""
    return sorted(int(round((j + 1) * (total - 1) / (k + 1))) for j in range(k))


def suggest_skip_sets(total: int = EXPECTED_BLOCKS) -> list[dict[str, Any]]:
    """Candidate index sets for the coordinator's experiment ladder.

    Every set keeps block 0 (the only block that sees the raw patch embedding; the
    residual stream has no other entry point) and block ``total-1`` (its output is
    what ``Head`` was trained on). Middle blocks of a pre-norm residual transformer
    are the usual candidates for removal because each contributes a small update to a
    stream that later blocks re-read; contiguous tails are the alternative hypothesis
    (late blocks refine detail the 2-step distilled sampler may not need). "Step 1
    only" variants halve both the saving and the risk: the high-noise pass that sets
    the mouth/pose structure stays intact and only the refinement pass is thinned.

    Each entry: ``name``, ``indices``, ``step``, ``rationale``, ``expected_dit_saving``
    (fraction of the two-step DiT time, i.e. k/30 for a drop, k/60 for one step),
    and ``cli`` (the run.py flags). Ordered cheapest-risk first within each family.
    """
    sets: list[dict[str, Any]] = []

    def add(name: str, indices: list[int], step: int | None, rationale: str) -> None:
        assert 0 not in indices and (total - 1) not in indices, name
        fraction = len(indices) / total / (1 if step is None else 2)
        cli = "--skip-dit-blocks " + " ".join(str(i) for i in indices)
        if step is not None:
            cli += f" --dit-skip-step {step}"
        sets.append({"name": name, "indices": list(indices), "step": step,
                     "rationale": rationale, "expected_dit_saving": round(fraction, 3), "cli": cli})

    for k in (3, 6, 9):
        middle = _evenly_spaced_middle(k, total)
        add(f"middle{k}_even", middle, None,
            f"{k} evenly spaced middle blocks, both steps: spreads the damage so no "
            f"consecutive stretch of the residual stream goes unrefined (~{k / total:.0%} of DiT)")
    for k in (3, 6, 9):
        tail = list(range(total - 1 - k, total - 1))
        add(f"tail{k}_before_last", tail, None,
            f"the {k} blocks just before the final one, both steps: tests the 'late blocks "
            f"only polish detail' hypothesis; the last block stays as the Head's input")
    for k in (3, 6, 9):
        middle = _evenly_spaced_middle(k, total)
        add(f"middle{k}_even_step1", middle, 1,
            f"same {k} middle blocks bypassed only at step 1 (t=625 refinement pass): "
            f"the structure pass at t=1000 runs the full depth; half the saving, less risk")
    add("tail6_before_last_step1", list(range(total - 7, total - 1)), 1,
        "6 late blocks bypassed only during refinement: cheapest way to test whether "
        "step 1 tolerates a shallower network at all")
    add("middle6_even_step0", _evenly_spaced_middle(6, total), 0,
        "control for the step-1 variant: the same blocks bypassed only at the high-noise "
        "step, to learn which pass is sensitive before spending a both-steps run")
    return sets


__all__ = [
    "CURRENT_STEP",
    "EXPECTED_BLOCKS",
    "install_dit_block_skip",
    "remove_dit_block_skip",
    "describe_dit_block_skip",
    "suggest_skip_sets",
]
