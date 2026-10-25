# 02 Fused kernel: forward parity

## Question

`mlx_upsample.kernels.bilinear` has one custom Metal kernel: it reads the 4 corner
pixels and writes 1 output value per thread, fused into a single GPU pass.
Does it match stock `mlx.nn.Upsample(mode="linear")`, for every workload and dtype?

## Method

- Kernel under test: `fused_uncached` in `mlx_upsample.kernels.bilinear` (`fused_uncached(x,
  scale_factor, align_corners)`), so `results/parity.csv` stays reproducible against the version
  that produced it. `tests/test_kernel.py` checks both `fused` (the cached default) and
  `fused_uncached`.
  Neither is in `mlx_upsample.variants.VARIANTS`: there is no backward pass yet, so neither
  takes part in the `01_crossover` sweep or gradient tests.
- Reference: `mlx.nn.Upsample(scale_factor=..., mode="linear", align_corners=...)`.
- Workloads (`mlx_upsample.workloads`): `grid()` and `real()`, same catalogs as `01_crossover`.
- For each workload: build the input with `make_input`, run both paths, and record shape, dtype,
  `max_abs_diff`, and `rel_err` (`max_abs_diff` divided by the largest absolute reference value).
- A row passes when shape and dtype match and `rel_err` is at most 1e-5, for both float32 and
  float16. Both paths compute in float32 internally from the same input values, so the expected
  difference is near float32 rounding, not a dtype-dependent error.
- `tests/test_kernel.py` checks the same kernel on a small, hand-picked case list (scales, an
  unequal height/width pair, odd sizes, small and large channel counts, both dtypes).

### Timing

- `sweep.py` times `fused` against the `01_crossover` variants (`gather`, `gather_compiled`,
  `matmul`, `pr4418`), forward pass only: `fused` has no backward pass yet, so there is no
  `fwd_bwd` row for it.
- Reuses `mlx_upsample.bench.run_sweep` for the timing loop, correctness gate, interleaved
  timing, and the resumable `results/sweep.jsonl`; see `01_crossover` for how that loop works.
- Workloads: `grid()` and `real()`, same catalogs as `01_crossover`, shuffled with the same
  seed (0).
- `analyze.py` drops invalid rows, then for each workload compares `fused` against each
  baseline and against the best of the non-fused variants. A method wins only if it is below
  0.9x the other time; otherwise the pair counts as a tie (same 10% noise margin as
  `01_crossover`).

## How to run

Run from any folder with the `cs6886-mlx-upsample` environment.

```sh
python experiments/02_fused_kernel/check.py --smoke   # about 10 workloads, writes results/scratch/
python experiments/02_fused_kernel/check.py           # full grid + real catalog, results/parity.csv
```

`check.py` writes `results/parity.csv` (one row per workload) and `results/parity_summary.md`
(pass counts and the worst `rel_err` per dtype). It does not measure time; see below for timing.

```sh
python experiments/02_fused_kernel/sweep.py --smoke     # about 40 rows, writes results/scratch/
python experiments/02_fused_kernel/sweep.py              # full grid + real catalog, results/sweep.jsonl
python experiments/02_fused_kernel/analyze.py            # reads results/sweep.jsonl by default
```

`sweep.py` writes `results/sweep.jsonl` and `results/meta.json`, and is resumable: a second run
only fills in missing (workload, variant) rows. `analyze.py` writes `results/speedup.csv`,
`results/figures/heatmap_fwd_{float32,float16}.png`, `results/figures/real_workloads.png`, and
`results/timing_summary.md`. Use `--in` on `analyze.py` to point at another `sweep.jsonl`, for example
the smoke-run output in `results/scratch/`.

## Findings

All 402 workloads pass: 201 float32 and 201 float16, across the `grid()` and `real()` catalogs.

Shape and dtype match stock in every case. A float16 input gives a float32 output, the same as
stock.

The worst relative error is 2.3e-07 for float32 (`depth_anything_s.fuse37`) and 2.0e-07 for
float16 (`depth_anything_l.fuse148`). Both values sit at float32 rounding level, about 40x under
the 1e-5 pass limit.

Note on the kernel design: `fused` is the minimal version for this step. Each thread reads only
the 2 real neighbor pixels per axis (4 corner pixels total) that have a nonzero weight, and
writes 1 output value. It uses no threadgroup memory, no tiling, and no backward pass. Matmul, in
contrast, multiplies by dense weight matrices that are mostly zeros.

Next step: measure the speed of `fused` against gather, matmul, and `pr4418`.

## Timing findings

The sweep ran 2010 rows on M1, forward pass only, with 0 invalid rows.

`fused` beats the best of `gather`, `gather_compiled`, `matmul`, and `pr4418` in 373 of 402
workloads. It ties in 28 and loses in 1. The median speedup over the best other variant is
2.52x (max 22.1x). Upscale cases get a median of 2.64x; downscale cases get 2.13x.

On the 56 real-model workloads, `fused` wins every case. Speedup ranges from 1.11x
(`pspnet.bin6`, batch 2, float32) to 9.1x (`rife.up2`, batch 8, float32). For the shapes from
issue #4435, `rife.up4` at batch 1, float16, is 3.2x faster than matmul and 15x faster than
gather.

The ties and the one loss (`grid.h128.s0.25.c4`, float32, 0.45 ms vs 0.35 ms) all share small
`C` (4) or small output size, under about 0.7 ms. At this size, a fixed per-call cost dominates
the kernel time. One hypothesis, not yet measured: `fused` rebuilds its index and weight tables
with several small MLX ops on every call, and this setup cost may be that fixed overhead.

See `results/figures/heatmap_fwd_float32.png`, `results/figures/heatmap_fwd_float16.png`, and
`results/figures/real_workloads.png` for the full picture.

## Fixed-cost check

### Question

The ties and the one loss above all sit at small output size, under about 0.7 ms. One
hypothesis: `fused` rebuilds its index and weight tables (`axis_tables`, which calls MLX's
`_linear_indices` and a few small ops) on every call, and this per-call setup cost is a large
part of the time at that size. Does caching the tables remove most of the gap?

### Method

- `mlx_upsample.kernels.bilinear.fused_cached` is the same kernel call as `fused`, but the six
  tables (`y_l, y_r, y_w, x_l, x_r, x_w`) come from a `functools.lru_cache`d helper keyed on
  `(H, W, scale_factor, align_corners)`. The helper calls `mx.eval` on the tables once, so later
  calls reuse GPU arrays that already exist.
- `experiments/02_fused_kernel/overhead.py` times, interleaved, for each workload: `fused`,
  `fused_cached` (cache warmed first), `tables_only` (builds the six tables and `mx.eval`s them,
  no kernel call), `floor` (one scalar add, the cheapest possible `mx.eval`'d GPU call), and the
  workload's best non-fused variant from `results/speedup.csv`.
- Workloads: every grid or real workload where `fused` did not win in the timing sweep (tie or
  loss), every real-model workload, and a size ladder (grid, `C=4`, float32, scale 2 and 0.25,
  all `H`), deduplicated.
- `fused_cached` output is checked against `gather` with the same tolerance as `bench.check`.
- Results are resumable in `results/overhead.jsonl` (meta in `results/overhead_meta.json`,
  separate from the timing sweep's `results/meta.json`).

### How to run

```sh
python experiments/02_fused_kernel/overhead.py --smoke     # about 5 workloads, writes results/scratch/
python experiments/02_fused_kernel/overhead.py              # full workload set, results/overhead.jsonl
python experiments/02_fused_kernel/overhead.py --analyze    # re-run analysis only, same input file
```

`overhead.py` writes `results/overhead.csv` (one row per workload: time for each variant, output
size, time saved by caching, `tables_only` share of `fused` time, speedup of `fused_cached` over
the best other variant, and win/tie/loss for both `fused` and `fused_cached`),
`results/overhead_summary.md`, and `results/figures/overhead.png` (time vs output size, log-log).

### Findings

The hypothesis holds. Caching saves about the net table cost: the net table cost is
`tables_only` time minus the launch `floor`, since `tables_only` is its own `mx.eval` call and
carries the same floor as every other variant. Median saved time (`fused` - `fused_cached`) is
0.105 ms vs a median net table cost of 0.126 ms, across all 103 workloads tested. On the 51
workloads where `fused` runs under 0.7 ms, the match is closer: 0.090 ms saved vs 0.096 ms net
table cost.

With cached tables, `fused_cached` beats the best other method on all 103 workloads tested,
including all 14 that tied or lost for uncached `fused` in this run.

The remaining fixed cost is the per-call launch floor, about 0.13-0.35 ms (median 0.23 ms). The
smallest `fused_cached` calls (about 0.15 ms) sit close to the smallest `floor` calls (about 0.13
ms), so caching removes most of the overhead that caching can remove.

Caveat: of the 29 ties/losses for uncached `fused` found in this experiment's own timing sweep
(`sweep.py` / `results/speedup.csv`), 13 were still ties or losses when re-timed here, with no
code change, and 16 flipped to wins. One workload that won in the sweep
(`grid.h64.s0.25.c4`, float32) flipped the other way and tied or lost here. So this rerun has 14
non-wins in total. Timing at these sub-millisecond sizes is noisy in both directions, so treat any
single win/tie/loss label near this size as approximate.

Note: after this check, `fused` is the cached kernel and `fused_uncached` is the version that
rebuilds tables every call. The rows named "fused" in `results/sweep.jsonl` and
`results/overhead.jsonl` were recorded with the uncached version (now `fused_uncached`); the
"fused_cached" rows used the cached version (now `fused`).

See `results/figures/overhead.png` for time vs output size, log-log, across all five variants.
