# Optional M5 MLX prefix reuse and GS64 prefill

This fork adds two independent, **default-off** flags:

- `--mlx-prefix-cache`: within one text-only causal `never` request, tokenize all
  questions and prefill their exact common token prefix once. Every question
  restores its own native KV/convolution/recurrent snapshot. No cross-request
  prefix lookup is performed. Instant, image and thinking requests bypass reuse.
- `--mlx-kernels gs64`: approximate INT8-activation prefill for eligible decoder
  linear/MLP modules. Weight files, embeddings, vision tower, answer-head readout,
  KV precision and the GDN recurrence stay unchanged. Sequences shorter than128
  tokens use the original linear math. Only the opted-in model is wrapped.

GS64 changes probabilities relative to the original W4A16 path. The sharing and
single-pass quantization/layout fusion preserve that GS64 result. Validate the
model and your decision boundaries before enabling it. Instant is refused for
this experimental kernel mode. The context guard applies to MLX with either flag.

`--mlx-kernels gs64-mlp` limits the same activation/packing optimization to
MLP modules. Attention and GDN projections use the original math. A fixed27B
calibration found this scope more faithful than full eligible-decoder GS64;
model-scale validation is still required.

## Pinned runtime

Tested on M5 Pro, macOS27, Python3.12, mlx0.32.4, mlx-vlm0.7.10 and
transformers5.20.2. The kernel feature refuses incompatible MLX versions rather
than silently falling back. Install your isolated environment with these versions:

```sh
python3.12 -m venv .venv
.venv/bin/pip install -e '.[mlx]' mlx==0.32.4 mlx-vlm==0.7.10 transformers==5.20.2
.venv/bin/python tools/setup_mlx_kernels.py
```

The setup tool downloads the official oMLX0.7.3 cp312 wheel, checks its fixed
SHA256 and extracts only the quantization kernel/runtime subset. It does not
install the oMLX server or run its initialization. Extracted binaries are ignored
by Git; the upstream Apache-2.0 license is retained. `--wheel PATH` reuses a local
copy. `STARTLUX_OMLX_RUNTIME` can locate the extracted subset outside the checkout.

```sh
.venv/bin/startlux-decide serve --backend mlx --model /path/to/AutoThink-MLX-4bit \
  --think-mode never --mlx-prefix-cache --mlx-kernels gs64-mlp \
  --host 0.0.0.0 --port 8091 --no-images --max-model-len 8192
```

Remove `--mlx-kernels gs64` for cache-only operation. Remove both flags for the
original computation path; no model conversion or weight restoration is needed.

## Tests and evidence

```sh
.venv/bin/python -m unittest tests.test_mlx_context_guard tests.test_mlx_prefix_pipeline tests.test_mlx_prefix_cache
STARTLUX_TEST_MLX_KERNELS=1 .venv/bin/python -m unittest tests.test_mlx_stage_native tests.test_mlx_kernel_policy
```

The native tests cover exact INT8 codes, FP32 scales and INT16 group sums against
the pinned binary, plus model-local routing and original single-token fallback.
Private article fixtures, labels, host configurations and raw responses are not
published. Model-scale validation results are summarized separately after testing.

See `NOTICE` for upstream provenance and license attribution.


## 27B validation (2026-11-05)

M5 Pro, macOS27, low-power mode throughout; official27B AutoThink affine-Q4,
never, None check, two questions per article, context8192. One resident model,
rotated/reversed execution order and0.5s untimed pauses; warmups excluded.

| Path | HTTP P50 | HTTP P95 | Sum of16 requests |
|---|---:|---:|---:|
| Original, no shared prefix | 9.998s | 16.036s | 174.33s |
| Prefix cache only | 6.121s | 9.202s | 104.46s |
| Prefix cache + gs64-mlp | 4.355s | 6.568s | 74.88s |

The selected path reduced summed HTTP time57.05% versus original, and28.32%
versus cache-only. All16 requests were faster than cache-only. These are request
latencies, including readout/service costs; the path generates no thought tokens.
The sum excludes pauses and is not a separately measured batching throughput.

All16 triage choices and all16 area choices matched original. Maximum absolute
probability delta was0.01038 versus original (including first-pass and None),
0.01400 versus cache-only. Seven manually interesting examples stayed Keep.
Repeated worst-delta requests and two interleaved requests reproduced outputs
exactly. Over-context input returned422. GPU-mean sampled peak60.21C,
fan998–1002RPM, experimental MLX allocation peak18.51GiB.

The full eligible-decoder gs64 profile was faster on a separate15-example run,
but changed one neutral near-tie from Drop to Keep; it was not selected for the27B
service. Keeping only the final4 or8 layers at original precision did not remove
that calibration change. MLP-only was selected on three calibration examples;
the final16 include13 examples not used for that scope choice. This is a small
compatibility validation, not a general accuracy guarantee or bitwise equivalence
to original inference. Decision prompts, thresholds and labels were not changed.

Regression checks:15 targeted/native tests passed;34 existing tests completed
with32 passed and2 skipped. Three real4B requests also reproduced the prior
full-GS64 fused experimental output exactly after packaging.
