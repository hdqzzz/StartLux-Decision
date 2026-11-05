import os
import unittest
if os.environ.get("STARTLUX_TEST_MLX_KERNELS") != "1":
    raise unittest.SkipTest("set STARTLUX_TEST_MLX_KERNELS=1 for pinned Apple native kernels")
import mlx.core as mx
from startlux_decide.backends.mlx_kernels import prepare_runtime
prepare_runtime()
from omlx.custom_kernels.quant_matmul import fast
from startlux_decide.backends._mlx_stage import stage_a

class FusedStageTests(unittest.TestCase):
 def test_codes_scales_group_sums_and_layout_are_exact(self):
  mx.random.seed(981)
  for dtype in [mx.bfloat16,mx.float16]:
   for shape in [(1,131,2560),(2,64,9216),(1,318,4096),(1,131,5120),(1,318,17408),(1,129,6144)]:
    for kind in ['normal','outlier','zero']:
     x=mx.random.normal(shape).astype(dtype)
     if kind=='outlier':x[...,0]=128
     if kind=='zero':x=mx.zeros(shape,dtype=dtype)
     reference=fast.oq_a8_stage_a_v8(x,1)
     out=stage_a(x,1,0)
     for key,a,b in zip(['qa','sa','ra'],reference,[out.qa,out.sa,out.ra]):
      delta=float(mx.max(mx.abs(a.astype(mx.float32)-b.astype(mx.float32))))
      self.assertEqual(delta,0,(dtype,shape,kind,key,delta))
  print('STAGE_A_ALL_EXACT',flush=True)

if __name__=='__main__':unittest.main(verbosity=2)
