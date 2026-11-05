# Extracted from official oMLX 0.7.3; only unused MoE/model installers removed.
# Native quantization and matmul formulas unchanged.
# ruff: noqa: N806
"""Model-local routing for Qwen INT8-activation prefill.

Eligible Q4/Q5/Q8 projections cache a dispatch plan and share activation
quantization where possible. Class wrappers are installed once, but only
modules tagged by an enabled model are routed. Importing this module installs
no patches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from omlx.patches.qwen35_packed_linear import PackedLinear

logger = logging.getLogger(__name__)

# Only Q4, Q5 and Q8 GS64 affine are routed; everything else is fallback.
_SUPPORTED_BITS = frozenset((4, 5, 8))
_GROUP_SIZE = 64

_PLAN_ATTR = "_omlx_oq_a8_plan"
_PREPARED_ATTR = "_omlx_oq_a8_prepared"

# Packed weights and Q8 metadata are reused; Q4/Q5 metadata is transposed once.
# Rowwise and GS64 activation scaling are supported independently per bit width.
_ACT_MODE_ROW = 0
_ACT_MODE_G64 = 1


# Prefill shorter than this stays on the existing path, where Stage A costs
# more than the faster matmul saves.
_MIN_TOKENS_DEFAULT = 128

# Attached by apply_qwen35_oq_a8_patch() to the modules of one loaded model.
# Its presence is the opt-in.
_CONFIG_ATTR = "_omlx_oq_a8_config"


@dataclass(frozen=True)
class OqA8Config:
    """One model's oQ A8 settings, as the engine passed them in."""

    min_tokens: int = _MIN_TOKENS_DEFAULT


def _config_for(module: Any) -> OqA8Config | None:
    """The configuration of the model that tagged ``module``, if any."""
    config = getattr(module, _CONFIG_ATTR, None)
    return config if config is not None and _kernels_available() else None






def _kernels_available() -> bool:
    try:
        from omlx.custom_kernels.quant_matmul import fast
    except ImportError:
        return False
    return fast.oq_a8_available()


@dataclass(frozen=True)
class OqA8Plan:
    """Frozen per-projection dispatch decision.

    Built once at classification time so the hot forward path never inspects
    quantization configuration.
    """

    bits: int
    group_size: int
    act_mode: int
    variant: int
    packed: bool = False

    @property
    def kernel(self) -> str:
        return f"q{self.bits}a8_g{self.group_size}"

    @property
    def layout(self) -> int:
        return _layout_for_bits(self.bits)


def _act_mode_for_bits(bits: int) -> int:
    if bits == 8:
        # GS64 scaling keeps activation outliers from setting the scale of a
        # whole row.
        return _ACT_MODE_G64
    return _ACT_MODE_ROW


# Q4/Q5 read Qa in Stage A v8's permuted K order. A Q8 byte is a whole code, so
# the Q8 kernel reads Qa in checkpoint order and no permuting copy is needed.
_LAYOUT_V8 = 0
_LAYOUT_NATURAL = 1


def _layout_for_bits(bits: int) -> int:
    return _LAYOUT_NATURAL if bits == 8 else _LAYOUT_V8


# 800-806 are the same kernel over different simdgroup grids. The field is
# flat -- within ~3% at M=2048 -- so these are the best of a flat field rather
# than a sharp optimum, and they differ by bit width because Q5 reads a wider
# window per row and prefers a smaller tile.
_DEFAULT_VARIANT_Q4 = 806
_DEFAULT_VARIANT_Q5 = 800

_VARIANT_MIN = 800

# Q8 picks its tile by row count and ignores the variant; both tiles are this wide.
_Q8_TILE_N = 64


def _variant_for_bits(bits: int) -> int:
    # Q4 and Q5 are autotuned independently: Q5 reads a second plane per
    # weight row and costs more registers, so the best tile need not match.
    # Q6 reads the same three-word window as Q5.
    return _DEFAULT_VARIANT_Q5 if bits in (5, 6) else _DEFAULT_VARIANT_Q4


def classify_linear(linear: Any) -> OqA8Plan | None:
    """Return the dispatch plan for ``linear``, or None if it is not eligible.

    The result is memoized on the module, so this runs once per projection
    per process however often the forward path asks for it.
    """
    cached = getattr(linear, _PLAN_ATTR, False)
    if cached is not False:
        return cached

    plan = _classify_uncached(linear)
    object.__setattr__(linear, _PLAN_ATTR, plan)
    return plan


def _classify_uncached(linear: Any) -> OqA8Plan | None:
    if isinstance(linear, PackedLinear):
        # Always Q4 GS64 with bfloat16 metadata; the kernel reads its tiles.
        return _plan_for(4, linear.output_dims, packed=True)
    if not isinstance(linear, nn.QuantizedLinear):
        return None
    if getattr(linear, "mode", None) != "affine":
        return None
    bits = getattr(linear, "bits", None)
    group_size = getattr(linear, "group_size", None)
    if bits not in _SUPPORTED_BITS or group_size != _GROUP_SIZE:
        return None
    if "bias" in linear:
        return None

    weight = getattr(linear, "weight", None)
    scales = getattr(linear, "scales", None)
    biases = getattr(linear, "biases", None)
    if weight is None or scales is None or biases is None:
        return None
    if weight.dtype != mx.uint32 or weight.ndim != 2:
        return None
    if scales.dtype not in (mx.float16, mx.bfloat16):
        return None
    if biases.dtype != scales.dtype or scales.shape != biases.shape:
        return None
    if scales.ndim != 2 or scales.shape[0] != weight.shape[0]:
        return None

    # The packed layout is the one MLX already stores and oMLX already
    # validates; no checkpoint conversion is required.
    input_dim = scales.shape[1] * _GROUP_SIZE
    if weight.shape[1] * 32 != input_dim * int(bits):
        return None
    return _plan_for(int(bits), weight.shape[0])


def _plan_for(bits: int, n: int, packed: bool = False) -> OqA8Plan | None:
    plan = OqA8Plan(
        bits=bits,
        group_size=_GROUP_SIZE,
        act_mode=_act_mode_for_bits(bits),
        variant=_variant_for_bits(bits),
        packed=packed,
    )

    from omlx.custom_kernels.quant_matmul import fast

    # N must tile exactly; the kernel refuses partial column tiles so the
    # weight decoder can stay bounds-check free.
    tile_bn = _Q8_TILE_N if bits == 8 else _variant_bn(plan.variant)
    if n % tile_bn != 0:
        logger.debug(
            "oq_a8: N=%d is not a multiple of BN=%d; leaving this projection "
            "on the existing path",
            n,
            tile_bn,
        )
        return None
    if not fast.oq_a8_available():
        return None
    return plan


# Must match oq_a8_nax_variant() in C++: (BM, BN, WM, WN), indexed from 800.
_VARIANT_TILES = {
    0: (64, 64, 2, 2),
    1: (128, 64, 4, 2),
    2: (64, 128, 2, 4),
    3: (128, 128, 4, 4),
    4: (32, 128, 1, 4),
    5: (256, 64, 8, 2),
    6: (32, 64, 1, 2),
}


def _variant_tile(variant: int) -> tuple[int, int, int, int]:
    try:
        return _VARIANT_TILES[variant - _VARIANT_MIN]
    except KeyError as exc:
        raise ValueError(f"Unknown oQ A8 tile variant {variant}") from exc


def _variant_bn(variant: int) -> int:
    return _variant_tile(variant)[1]


def _prepared_weights(linear: Any):
    """Cache transposed Q4/Q5 metadata; Q8 and PackedLinear are read as stored."""
    if not isinstance(linear, PackedLinear) and linear.bits == 8:
        return (linear.weight, linear.scales, linear.biases)
    cached = getattr(linear, _PREPARED_ATTR, None)
    if cached is not None:
        return cached

    if isinstance(linear, PackedLinear):
        prepared = (
            linear.packed_weight.view(mx.uint32).reshape(linear.output_dims, -1),
            linear.packed_scales,
            linear.packed_biases,
        )
    else:
        prepared = (
            linear.weight,
            mx.contiguous(linear.scales.T),
            mx.contiguous(linear.biases.T),
        )
    mx.eval(*prepared)
    object.__setattr__(linear, _PREPARED_ATTR, prepared)
    return prepared


@dataclass(frozen=True)
class StageA:
    """Quantized activation shared across projections of one input.

    Stage A runs once per activation. The MLP feeds it to gate and up, and a
    linear-attention block feeds it to qkv/z/a/b, whose bit widths may differ
    -- mixed bit widths need no separate activation quantization.
    """

    qa: mx.array
    sa: mx.array
    ra: mx.array
    act_mode: int
    layout: int = _LAYOUT_V8


def stage_a(
    x: mx.array, act_mode: int = _ACT_MODE_ROW, layout: int = _LAYOUT_V8
) -> StageA:
    """Quantize one activation into the layout the kernel reads.

    For Q4/Q5, Qa carries the schedule's within-group K order, and Ra (and Sa
    in per-group mode) come back group-major to match the weight metadata. Q8
    keeps Qa in checkpoint order and only transposes the metadata. Qa is
    rebuilt on every prefill, which uses temporary activation memory without
    changing the checkpoint.
    """
    from omlx.custom_kernels.quant_matmul import fast

    if layout == _LAYOUT_NATURAL:
        qa, sa, ra = fast.oq_a8_stage_a_natural(x, act_mode)
    else:
        qa, sa, ra = fast.oq_a8_stage_a_v8(x, act_mode)
    return StageA(qa=qa, sa=sa, ra=ra, act_mode=act_mode, layout=layout)


def apply_plan(linear: Any, stage: StageA, plan: OqA8Plan) -> mx.array:
    """Run one projection against an already-quantized activation."""
    if stage.act_mode != plan.act_mode:
        raise ValueError(
            "Stage-A activation mode "
            f"{stage.act_mode} does not match the projection's plan "
            f"({plan.act_mode}); quantize once per activation policy."
        )
    if stage.layout != plan.layout:
        raise ValueError(
            f"Stage-A layout {stage.layout} does not match the projection's "
            f"plan ({plan.layout}); quantize once per kernel layout."
        )
    from omlx.custom_kernels.quant_matmul import fast

    weight, scales, biases = _prepared_weights(linear)
    return fast.oq_a8_qmm_t(
        stage.qa,
        stage.sa,
        stage.ra,
        weight,
        scales,
        biases,
        plan.bits,
        plan.act_mode,
        plan.variant,
        packed=plan.packed,
    )


def oq_a8_linear(linear: Any, x: mx.array) -> mx.array:
    """Single projection with no shared activation; falls back transparently."""
    config = _config_for(linear)
    plan = classify_linear(linear) if config is not None else None
    if plan is None or not _shape_eligible(x, config):
        return linear(x)
    return apply_plan(linear, stage_a(x, plan.act_mode, plan.layout), plan)


# Below this many rows the Stage-A quantization pass costs more than the
# faster GEMM saves. Measured on a 5120x8704 Q4 MLP block, the crossover sits
# between 4 and 8 rows and the win is already 1.6x by 128. The default is set
# well above the crossover rather than at it, so that decode and speculative
# verify shapes -- which are latency-critical and only a handful of rows --
# stay on the existing path even where a caller does not flag them.


def _shape_eligible(x: mx.array, config: OqA8Config) -> bool:
    # Decode is out of scope for version 1; the kernel is built for
    # prefill and the tiles start at 32 rows.
    if x.ndim < 2 or x.dtype not in (mx.float16, mx.bfloat16):
        return False
    if x.shape[-2] <= 1 or x.shape[-2] < config.min_tokens:
        return False
    return x.shape[-1] % _GROUP_SIZE == 0


def oq_a8_mlp(mlp: Any, x: mx.array, activation) -> mx.array | None:
    """SwiGLU MLP with Stage A shared between gate and up.

    Gate and up share Stage A when their activation scaling policies match.
    Returns None when any of the three projections is not eligible, leaving
    the caller on its existing path.
    """
    config = _config_for(mlp)
    if config is None or not _shape_eligible(x, config):
        return None

    gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
    gate_plan = classify_linear(gate)
    up_plan = classify_linear(up)
    down_plan = classify_linear(down)
    if gate_plan is None or up_plan is None or down_plan is None:
        return None
    if (gate_plan.act_mode, gate_plan.layout) != (up_plan.act_mode, up_plan.layout):
        return None

    shared = stage_a(x, gate_plan.act_mode, gate_plan.layout)
    g = apply_plan(gate, shared, gate_plan)
    u = apply_plan(up, shared, up_plan)
    h = activation(g, u)

    if not _shape_eligible(h, config):
        return down(h)
    return apply_plan(down, stage_a(h, down_plan.act_mode, down_plan.layout), down_plan)


def oq_a8_gdn_projections(
    gdn: Any,
    x: mx.array,
    names: tuple[str, ...] = ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"),
) -> dict[str, mx.array] | None:
    """Linear-attention input projections sharing one Stage A.

    qkv is Q4 while z/a/b may be Q5, and that mix is exactly the case the
    shared activation is designed for. Returns None unless every
    requested projection is eligible under one activation policy.
    """
    config = _config_for(gdn)
    if config is None or not _shape_eligible(x, config):
        return None

    plans: dict[str, OqA8Plan] = {}
    for name in names:
        linear = getattr(gdn, name, None)
        if linear is None:
            return None
        plan = classify_linear(linear)
        if plan is None:
            return None
        plans[name] = plan

    policies = {(plan.act_mode, plan.layout) for plan in plans.values()}
    if len(policies) != 1:
        # A split policy would need one Stage A per policy, which costs more
        # than the mixed dispatch saves on the shapes measured here.
        return None

    shared = stage_a(x, *policies.pop())
    return {
        name: apply_plan(getattr(gdn, name), shared, plan)
        for name, plan in plans.items()
    }


