import os
import unittest
if os.environ.get("STARTLUX_TEST_MLX_KERNELS") != "1":
    raise unittest.SkipTest("set STARTLUX_TEST_MLX_KERNELS=1 for pinned Apple native kernels")
import unittest
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_5 import language as lang
from startlux_decide.backends.mlx_kernels import KernelPolicy, prepare_runtime
prepare_runtime()
from startlux_decide.backends import _mlx_a8 as a8


class KernelPolicyTest(unittest.TestCase):
    def test_local_wrapper_matches_gs64_reference_and_closes(self):
        mx.random.seed(981)
        m = lang.Qwen3_5MLP(256, 512); m.set_dtype(mx.bfloat16)
        nn.quantize(m, bits=4, group_size=64)
        other = lang.Qwen3_5MLP(256,512)
        x = mx.random.normal((1,131,256)).astype(mx.bfloat16)
        original_mlp, original_linear = type(m), type(m.gate_proj)
        policy = KernelPolicy(m)
        try:
            self.assertIs(type(other), original_mlp)
            plans=[a8.classify_linear(y) for y in [m.gate_proj,m.up_proj,m.down_proj]]
            stage=a8.stage_a(x,1,plans[0].layout)
            h=lang.swiglu(a8.apply_plan(m.gate_proj,stage,plans[0]),a8.apply_plan(m.up_proj,stage,plans[1]))
            reference=a8.apply_plan(m.down_proj,a8.stage_a(h,1,plans[2].layout),plans[2])
            got=m(x)
            self.assertEqual(float(mx.max(mx.abs(reference-got))),0)
            self.assertGreater(policy.counts['stage_a_fused'],0)
            policy.reset()
            self.assertIsNone(policy.last)
            # Single-token/decode work must use the original projection math.
            short=x[:,:1]
            want=original_mlp.__call__(m,short)
            actual=m(short)
            self.assertEqual(float(mx.max(mx.abs(want-actual))),0)
        finally:
            policy.close()
        self.assertIs(type(m),original_mlp)
        self.assertIs(type(m.gate_proj),original_linear)
        self.assertIsNone(policy.last)
