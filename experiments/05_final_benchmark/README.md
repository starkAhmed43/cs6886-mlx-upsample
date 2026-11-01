# 05 Final Benchmark: the user-facing nn.Upsample path, real-model catalog

## Question

Earlier experiments time individual variants directly
(`mlx_upsample.variants.gather`/`matmul`/`pr4418_shipped`/`fused`). Does the fused kernel still
win when a user calls it the normal way, through `nn.Upsample(mode="linear")`, on the full
real-model catalog (`workloads.real()`, 56 workloads), for both forward and forward+backward?

Three implementations:

- **stock**: `nn.Upsample` in the pip-mlx env (`cs6886-mlx-upsample`, mlx 0.32.4). Uses MLX's
  gather-based linear path (PR #4418 is not merged).
- **pr4418**: `mlx_upsample.variants.pr4418_shipped`, the PR as shipped, same env as stock. Stock
  MLX does not include the PR, so this variant calls the PR's code directly.
- **fused_cpp**: `nn.Upsample` in the from-source mlx env (`cs6886-mlx-dev`, branch
  `upsample-fused` of `~/GitHub/mlx`). In this env, `nn.Upsample`'s linear path for 4-D inputs
  dispatches to `mx.fast.upsample_bilinear`, a C++/Metal primitive with a differentiable VJP.

## Method

- A small adapter `f(x, scale_factor, align_corners)` wraps `nn.Upsample`: the layer is built
  once per workload (`nn.Upsample(scale_factor=..., mode="linear", align_corners=...)`), and the
  adapter just calls it. `pr4418_shipped` already has this signature.
- Each (env, workload, run) is timed in its own fresh subprocess, using that env's python
  (`experiments/01_crossover/confirm.py`'s pattern), so GPU/driver state never leaks across runs.
  One worker times every variant that lives in its env (`stock` + `pr4418` together in the pip
  env, `fused_cpp` alone in the dev env) for both `fwd` and `fwd_bwd`, interleaved and shuffled
  with `mlx_upsample.bench.time_interleaved` (seeded by the run index).
- The orchestrator loops run index outermost, workload next, and alternates which env goes first
  based on `(workload index + run) % 2`, so each workload's env order flips between runs and
  timing drift (thermal, clock) spreads evenly across variants and envs instead of always
  favoring whichever ran first.
- Correctness: output and gradient are checked against the float32
  `mlx_upsample.variants.gather` reference with `mlx_upsample.bench.check`. `out_dtype` is
  recorded for every row. `fused_cpp` returns float16 output for float16 input, while `stock` and
  `pr4418` upcast to float32 — this is expected (`mx.fast.upsample_bilinear` preserves the input
  dtype) and is not a validity failure; only the numeric error is gated (1e-5 for float32, 1e-2
  for float16 forward; 1e-5 relative gradient error for float32, recorded but not gated for
  float16).
- Rows are appended to `results/runs.jsonl` as they complete
  (`mlx_upsample.bench.ResultStore`, subclassed so the resumability key also includes the run
  index), so a crash only loses the in-flight row and a rerun is a no-op.
- `--max-mb` (default 256) caps workloads by output size, same meaning as `03_tiling`/`04_backward`.
- Workers run strictly one at a time; never in parallel (this is GPU timing).

## How to run

Run from the repo root with the `cs6886-mlx-upsample` env's python (the orchestrator calls out to
`cs6886-mlx-dev`'s python itself for the `fused_cpp` rows):

```sh
python experiments/05_final_benchmark/run.py --smoke   # 3 small workloads x 2 runs, results/scratch/
python experiments/05_final_benchmark/run.py            # full catalog x 10 runs, results/runs.jsonl
python experiments/05_final_benchmark/analyze.py --smoke
python experiments/05_final_benchmark/analyze.py        # reads results/runs.jsonl by default
```

`run.py` writes `results/runs.jsonl` and `results/meta.json` (environment metadata for both envs,
plus the `~/GitHub/mlx` commit and whether that tree is dirty). `analyze.py` writes
`results/table.csv` (per workload/variant/pass: median ms over runs, run-to-run spread, speedup
vs stock, vs pr4418, win/tie/loss), `results/summary.md`, and `results/figures/speedup.png`.

## Findings

53 of the 56 real-model workloads fit under the 256 MB output cap. Each ran 10 fresh-process
runs per variant and pass, for 3180 rows total, with 0 invalid rows. All numbers below come from
`results/table.csv` and `results/summary.md`.

### fused_cpp wins on every workload

| pass | geomean vs stock | geomean vs pr4418 | median vs stock | median vs pr4418 | min vs stock | min vs pr4418 | max vs stock |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fwd | 16.6x | 5.3x | 15.3x | 3.9x | 4.5x | 1.30x | 47.1x |
| fwd_bwd | 13.9x | 4.7x | 13.1x | 3.7x | 5.6x | 0.99x | 37.1x |

Win/tie/loss vs stock: 53/0/0 for both passes. Vs pr4418: `fwd` is 53/0/0; `fwd_bwd` is 51 wins,
2 ties, 0 losses. Across both passes, fused_cpp beats pr4418 in 104 of 106 comparisons, ties the
other 2, and never loses.

The largest gain is `depth_anything_l.fuse148` (B=1, 148x148x256, scale x2, align_corners=True,
float16), forward pass: stock 75.6 ms, pr4418 75.8 ms, fused_cpp 1.6 ms, a 47x speedup. The
smallest gain vs stock is `upernet.fpn16` (B=1, 16x16x512, scale x2, float16), forward pass:
1.33 ms vs 0.293 ms, a 4.5x speedup. Small tensors leave less room to win, since a fixed per-call
cost dominates the total time.

### The gain is not only from the float16 output

fused_cpp keeps float16 output when the input is float16. stock and pr4418 upcast their output to
float32 (see the dtype table in `summary.md`). A smaller output means less memory traffic, so part
of the gain could come from this alone. But float32 inputs, where all three variants output
float32, show gains of the same size:

| pass | dtype | n | geomean vs stock | geomean vs pr4418 |
| --- | --- | --- | --- | --- |
| fwd | float16 | 28 | 16.8x | 5.7x |
| fwd | float32 | 25 | 16.5x | 4.9x |
| fwd_bwd | float16 | 28 | 13.4x | 4.8x |
| fwd_bwd | float32 | 25 | 14.6x | 4.6x |

The output dtype change is a deliberate behavior change of the MLX branch, not the main source of
the speedup. Users of the branch must know about it.

### Gradients

Max relative gradient error (forward+backward pass), from `summary.md`:

- float32: fused_cpp 2.3e-6, pr4418 2.56e-6, stock 1.84e-6. All three sit at float32 rounding
  level.
- float16: fused_cpp 6.2e-4, pr4418 1.6e-3, stock 2.7e-2. fused_cpp is about 44x better than
  stock. The cause matches `04_backward`: the fused kernel accumulates each input gradient in
  float32 and rounds to float16 once, at the end.

### The 2 ties: PSPNet backward at large scales

Both ties are forward+backward, B=1, float16, C=512, align_corners=True:

| workload | input | scale | stock | pr4418 | fused_cpp | vs pr4418 |
| --- | --- | --- | --- | --- | --- | --- |
| pspnet.bin1 | 1x1 | x60 | 14.4 ms | 1.64 ms | 1.65 ms | 0.99x |
| pspnet.bin2 | 2x2 | x30 | 20.9 ms | 1.65 ms | 1.64 ms | 1.01x |

fused_cpp still beats stock by 8.7x and 12.8x here. A likely cause: the backward kernel runs one
GPU thread per input value, and these inputs are tiny (1x1 or 2x2 at 512 channels, so 512 or 2048
threads), while each thread loops serially over a 60x60 or 30x30 output window. Too few threads
leave the GPU mostly idle. PR #4418's matmul path handles this shape well, since its weight
matrices are tiny for a small input. A possible fix, not done here, is to split each thread's
output loop across several threads and reduce the partial sums. The other PSPNet rows also show
the lowest forward+backward speedups vs pr4418, between 1.16x and 1.30x.

### Noise and caveats

- Run-to-run spread, `(max-min)/median` over 10 runs, has a median of 14% across all rows and a
  max of 65% (`depth_anything_l.fuse74`, B=1, float16, forward pass, fused_cpp, a 0.55 ms row).
  Taking the median over 10 fresh processes, plus the 0.9x tie margin, absorbs this noise. Besides
  the 2 PSPNet ties, the closest rows to the margin are two other PSPNet fwd_bwd rows at 1.16x and
  1.17x vs pr4418.
- First-call cost: the primitive compiles one Metal pipeline per new shape, since sizes are
  function constants. This costs about 6 ms, once per shape. Warm-up excludes this cost from the
  timings above; it matters for workloads with many distinct shapes.
- The fused_cpp numbers come from an uncommitted MLX tree on v0.32.4 (commit 36969c094,
  `mlx_repo_dirty: true` in `meta.json`). The exact diff is `results/mlx_uncommitted.patch`. To
  reproduce: check out v0.32.4, apply the patch, and build MLX from source into the dev env.
- Measured on one machine only: Apple M1, 16 GB, macOS 27.2.

See `results/figures/speedup.png` for the full distribution.
