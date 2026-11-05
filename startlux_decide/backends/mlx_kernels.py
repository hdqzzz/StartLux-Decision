"""Opt-in GS64 W4A8 prefill on M5; model-local wrappers, original readout head.

The INT8 activation path is approximate. Sharing and layout fusion preserve
its quantized result. Default backend behavior remains unchanged.
"""
from collections import Counter
from dataclasses import replace
import importlib.metadata as md
import os
from pathlib import Path
import sys


def prepare_runtime():
    path = Path(os.environ.get("STARTLUX_OMLX_RUNTIME",
                               Path(__file__).resolve().parents[2] / "third_party/omlx-runtime"))
    if not path.is_dir():
        raise RuntimeError("GS64 kernels need the pinned runtime: run tools/setup_mlx_kernels.py")
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    from omlx.custom_kernels.quant_matmul import fast
    if not fast.oq_a8_available():
        raise RuntimeError(f"oMLX INT8 kernels unavailable: {fast.import_error()}")


class KernelPolicy:
    def __init__(self, body, mlp_only=False):
        import mlx.core as mx
        if mx.__version__ != "0.32.4" or md.version("mlx-vlm") != "0.7.10":
            raise RuntimeError("GS64 kernels are pinned to mlx0.32.4 and mlx-vlm0.7.10")
        prepare_runtime()
        import mlx.nn as nn
        from mlx_vlm.models.qwen3_5 import language as lang
        from . import _mlx_a8 as a8
        self.last = None
        self.enabled = True
        self.counts = Counter()
        self.changed = []
        owner = self

        class Linear(nn.QuantizedLinear):
            def __call__(module, x):
                plan = a8.classify_linear(module)
                if owner.enabled and plan is not None and a8._shape_eligible(x, a8.OqA8Config()):
                    owner.counts["a8_linear"] += 1
                    return a8.apply_plan(module, owner.stage(x, plan), plan)
                return nn.QuantizedLinear.__call__(module, x)

        class MLP(lang.Qwen3_5MLP):
            def __call__(module, x):
                projections = [module.gate_proj, module.up_proj, module.down_proj]
                plans = [a8.classify_linear(m) for m in projections]
                if owner.enabled and all(p is not None for p in plans) and a8._shape_eligible(x, a8.OqA8Config()):
                    gate, up, down = projections
                    gp, up_plan, dp = plans
                    shared = owner.stage(x, gp)
                    g = a8.apply_plan(gate, shared, gp)
                    u = a8.apply_plan(up, shared, up_plan)
                    h = lang.swiglu(g, u)
                    owner.counts["a8_mlp"] += 1
                    owner.counts["mlp_gate_up_shared"] += 1
                    return a8.apply_plan(down, owner.stage(h, dp), dp)
                return lang.Qwen3_5MLP.__call__(module, x)

        # Materialize before changing classes: named_modules need not understand wrappers.
        root_is_mlp = isinstance(body, lang.Qwen3_5MLP)
        for name, module in list(body.named_modules()):
            if mlp_only and not root_is_mlp and not (isinstance(module, lang.Qwen3_5MLP) or ".mlp." in name):
                continue
            original = type(module)
            if isinstance(module, nn.QuantizedLinear):
                plan = a8.classify_linear(module)
                if plan is None:
                    continue
                object.__setattr__(module, a8._PLAN_ATTR, replace(plan, act_mode=1))
                target = Linear
            elif isinstance(module, lang.Qwen3_5MLP):
                target = MLP
            else:
                continue
            self.changed.append((module, original))
            object.__setattr__(module, "__class__", target)

    def reset(self):
        self.last = None

    def stage(self, x, plan):
        from . import _mlx_a8 as a8
        from ._mlx_stage import stage_a
        previous = self.last
        if previous is not None and previous[0] is x and previous[1:3] == (plan.act_mode, plan.layout):
            self.counts["stage_a_reused"] += 1
            return previous[3]
        if plan.act_mode == 1 and plan.layout == 0:
            result = stage_a(x, plan.act_mode, plan.layout)
            self.counts["stage_a_fused"] += 1
        else:
            result = a8.stage_a(x, plan.act_mode, plan.layout)
        self.last = (x, plan.act_mode, plan.layout, result)
        return result

    def close(self):
        self.reset()
        for module, original in reversed(self.changed):
            object.__setattr__(module, "__class__", original)
        self.changed.clear()
