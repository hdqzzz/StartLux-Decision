"""GS64 quantization + v8 permutation + metadata transpose in one pass.

Adapted from oMLX oq_a8.metal, fixed commit2165e757; parent OMLX-LICENSE.
Preserves its FP32 scale, ties-to-even rounding and INT16 group sums.
"""
from functools import lru_cache
import mlx.core as mx
from . import _mlx_a8 as a8

SOURCE=r'''
const int row=int(threadgroup_position_in_grid.x);
const int groups=K/64;
const int M=x_shape[0];
const int lane=int(thread_index_in_simdgroup);
const int sg=int(simdgroup_index_in_threadgroup);
for(int g=sg;g<groups;g+=8){
    const int k0=g*64+lane;
    const int k1=k0+32;
    const float x0=float(x[size_t(row)*K+k0]);
    const float x1=float(x[size_t(row)*K+k1]);
    const float amax=simd_max(max(abs(x0),abs(x1)));
    const float inv=amax>0.0f?(127.0f/amax):0.0f;
    const int q0=int(clamp(rint(x0*inv),-127.0f,127.0f));
    const int q1=int(clamp(rint(x1*inv),-127.0f,127.0f));
    // Inverse of native v8 slot16c+4t+j -> k16c+8(t>>1)+2j+(t&1).
    const int p0=(k0/16)*16+((k0&8))+((k0&1)*4)+((k0&6)/2);
    const int p1=(k1/16)*16+((k1&8))+((k1&1)*4)+((k1&6)/2);
    qa[size_t(row)*K+p0]=char(q0);
    qa[size_t(row)*K+p1]=char(q1);
    const int sum=simd_sum(q0+q1);
    if(lane==0){
        sa[size_t(g)*M+row]=amax>0.0f?(amax/127.0f):0.0f;
        ra[size_t(g)*M+row]=short(sum);
    }
}
'''

@lru_cache(maxsize=1)
def kernel():
    return mx.fast.metal_kernel(name='phase2_gs64_stagea_v8',input_names=['x'],
        output_names=['qa','sa','ra'],source=SOURCE)


def stage_a(x,act_mode,layout):
    assert act_mode==1 and layout==0
    shape=x.shape;k=shape[-1];rows=x.size//k
    qa,sa,ra=kernel()(inputs=[x.reshape(rows,k)],template=[('K',k)],
        grid=(rows*256,1,1),threadgroup=(256,1,1),
        output_shapes=[shape,(k//64,rows),(k//64,rows)],
        output_dtypes=[mx.int8,mx.float32,mx.int16])
    return a8.StageA(qa,sa,ra,act_mode,layout)
