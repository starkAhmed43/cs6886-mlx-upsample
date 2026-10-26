# 04 Backward: a hand-written VJP for the fused kernel

## Question

The fused Metal kernel (`02_fused_kernel`, `03_tiling`) has no backward pass, so MLX cannot
train through it: `mx.grad` falls back to differentiating the Python wrapper, which does not
exist for a custom kernel without a VJP. Can a hand-written backward kernel make the fused
kernel usable for training? Is forward+backward together still faster than stock MLX
(`gather`, `matmul`, `pr4418`)?

## Method

- Kernel under test: `upsample(x, scale_factor, align_corners)` in
  `mlx_upsample.kernels.bilinear`, a differentiable wrapper around the forward kernel with a
  custom VJP. The backward kernel runs gather-form: one GPU thread per input value, which loops
  over the contiguous range of output pixels that read it and accumulates their gradient
  contribution. No atomics are needed, since each thread only ever writes its own input
  gradient, so results are deterministic run to run.
- Baselines: `gather`, `matmul`, `pr4418` (stock-equivalent variants from
  `mlx_upsample.variants`), each differentiated by MLX's own autograd.
- Correctness: every variant's forward output and gradient are checked against a float32
  `gather` reference (`mlx_upsample.bench.check`). float32 gradients are gated at a relative
  error of 1e-5 or less; a row that fails is marked invalid. float16 gradient error is recorded
  but not gated, since float16 accumulation error is expected.
- Unlike `03_tiling` (forward only, upscale only, since tiling only has reuse to exploit when
  scale > 1), this sweep covers every scale in `grid()` and `real()`, both upscale and
  downscale, since the backward kernel has no such restriction.
- Timing: `mlx_upsample.bench.run_sweep`, pass `"fwd_bwd"` only, interleaved timing against all
  four variants at once. Memory: `--max-mb` (default 256 MB) caps output size, same as
  `03_tiling`. `mx.set_cache_limit(1 << 30)` returns freed buffers to the OS between workloads.
- Timing rule: a method wins only if it is below 0.9x the other time; otherwise the pair counts
  as a tie (same 10% noise margin as the earlier experiments).

## How to run

Run from any folder with the `cs6886-mlx-upsample` environment.

```sh
python experiments/04_backward/sweep.py --smoke     # a few rows, writes results/scratch/
python experiments/04_backward/sweep.py             # full grid + real catalog, results/sweep.jsonl
python experiments/04_backward/analyze.py           # reads results/sweep.jsonl by default
python experiments/04_backward/analyze.py --in experiments/04_backward/results/scratch/sweep.jsonl --out-dir experiments/04_backward/results/scratch
python experiments/04_backward/retime.py            # re-times the 2 losses in isolation, writes results/retime.jsonl
```

`sweep.py` writes `results/sweep.jsonl` and `results/meta.json`, and is resumable: a second run
only fills in missing (workload, variant) rows. `analyze.py` writes `results/speedup.csv`,
`results/summary.md`, `results/figures/heatmap_fwd_bwd_{float32,float16}.png`, and
`results/figures/real_workloads.png`.

## Findings

### Gradients are correct

All gradient checks pass. There are 0 invalid gradient rows.

- float32: fused's worst relative error is 2.07e-6, close to stock gather's 1.26e-6.
- float16: fused's worst relative error is 4.85e-4, 50x better than stock gather's 2.41e-2.
  Fused accumulates each input gradient in float32 and rounds to float16 once at the end.
  A likely cause for stock gather: its backward scatter-adds into a float16 gradient array, so
  rounding error grows with the number of output pixels that read each input pixel. This fits
  the worst stock rows, the PSPNet bins with the largest scales (bin2, x30: 2.41e-2).

Five kinds of tests back this up: parity with the stock kernel, float16 accuracy, an adjoint
check (`<Jx, y> == <x, J^T y>`), finite differences, and the PSPNet 1x1 -> 60x60 case.

### Fused wins most of the time

Against the best stock variant at each point (the faster of `gather`, `matmul`, `pr4418`):
fused wins 376 of 380 cases, ties 2, and loses 2. Median speedup is 3.49x, max 25.5x.

On the real-model catalog alone: 50 wins, 2 ties, 1 loss out of 53. Median speedup 3.24x.
Downscale cases have a median speedup of 3.71x; upscale cases 3.34x.

Some real examples (forward+backward, `fused` ms vs `best_stock` ms, from `speedup.csv`):

| workload | B | dtype | fused | best stock | speedup |
| --- | --- | --- | --- | --- | --- |
| rife.down2 | 8 | float32 | 6.00 ms | 60.3 ms | 10.0x |
| depth_anything_l.fuse148 | 1 | float16 | 4.44 ms | 26.8 ms | 6.03x |
| segformer.logits | 1 | float16 | 6.53 ms | 25.4 ms | 3.90x |
| upernet.logits | 1 | float16 | 6.42 ms | 25.5 ms | 3.98x |

### Why the gain is bigger than forward alone

The forward-only sweep (`02_fused_kernel`) found a median speedup of 2.52x (max 22.1x).
Forward+backward together reaches a median of 3.49x. A likely reason: stock MLX must
differentiate through its own gather or matmul ops. A gather's backward is a scatter-add, and
this likely builds full-size temporary arrays along the way. The fused kernel's backward is a
single hand-written pass: it reads the incoming gradient once and writes the input gradient
once, with no intermediate arrays.

### The 2 losses

**grid.h1536.s0.25.c256, float32, is a memory-pressure artifact.** In the sweep, this workload
times fused at 2283 ms against matmul's 1354 ms (a loss), because the sweep keeps four
variants' buffers and a float32 `gather` reference alive at once: 2.4 GB input, 2.4 GB gradient,
plus the reference output. On a 16 GB machine this likely caused memory pressure and swap.

`retime.py` re-times only `fused` and `matmul`, in a fresh process, with no reference built.
Result (`results/retime.jsonl`): fused 182.0 ms vs matmul 772.1 ms, a 4.2x win for fused. The
sweep's loss here is not a real kernel problem; it is likely swap pressure from running too many
variants and a reference at once.

**pspnet.bin1, B=1, float16, is a real loss.** This workload upsamples a 1x1x512 input to
60x60x512 (align_corners, scale 60x). `retime.py` confirms it in isolation: fused 1.67 ms vs
matmul 1.06 ms. The cause is the kernel's grid shape: with only 1x1 input pixels, fused's
backward launches one thread per (input pixel, channel), so only 512 threads run here, each one
looping serially over all 3600 output pixels. Most of the GPU's compute units sit idle. The
absolute cost is small, about 0.6 ms, so this loss does not matter in practice. A possible fix,
not implemented here, is to split each thread's output-row loop across more threads when the
input is this small.

`pspnet.bin2` (a 2x2 input, scale 30x) ties for a similar reason: fused's backward launches 4
input pixels x 512 channels = 2048 threads. Measured backward range tables show each input index
along an axis is read by 59 of the 60 outputs, so each thread loops over 59x59 = 3481 output
pixels, about as long as bin1's 3600-pixel loop. But bin2 has 4x more threads than bin1, so the
gap to matmul is smaller.

### 5 invalid rows

All 5 are `grid.h2048.s0.25.c256`: every float32 variant, and the float16 `matmul` variant, ran
out of GPU memory (OOM). The result cap in `sweep.py` (`--max-mb`) only bounds output bytes; for
a downscale (scale 0.25) the *input* is 16x larger than the output, so a workload can pass the
output cap and still exhaust memory on the input and its gradient.

### Limits

- Measured on one Apple M1, 16 GB. Results may differ on other chips or memory sizes.
- Timings are forward+backward together, not backward alone.
- Both the fused forward and backward kernels use cached lookup tables keyed by shape; the
  first call for a new shape pays the cost of building that table.
- The kernel uses 32-bit indexing, so it is not validated on tensors needing 64-bit offsets.
