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
  --think-mode never --mlx-prefix-cache --mlx-kernels gs64 \
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
