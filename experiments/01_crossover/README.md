# 01 Crossover: gather or matmul for bilinear upsampling

## Question

PR #4418 uses matmul only when it upscales and the input is at most 128 px. Nobody measured that
limit. Where is matmul really faster than gather, forward and backward? How much time does the PR
gate lose?

## Method

- Variants (`mlx_upsample.variants`): `gather`, `gather_compiled`, `matmul`, `pr4418` (the PR as shipped).
- Workloads (`mlx_upsample.workloads`):
  - `grid`: H = W in 32..1024, scale in {0.25, 0.5, 1.5, 2, 3, 4}, C in {4, 32, 256}, float32 and float16, B = 1.
  - `real`: bilinear calls from RIFE, SegFormer, UPerNet, DeepLabV3+, PSPNet, Depth Anything and SAM.
  - Workloads with a float32 output above 512 MB are skipped.
- Timing: all variants of one workload and pass are timed together with `time_interleaved`: 3 warmup calls each,
  then 7 repeats of about 50 ms each, with the variant order shuffled in every repeat. All variants see the same machine state.
  `ms_median` is the median repeat. Each call evaluates its result.
  The `fwd_bwd` pass uses `mx.vjp` with a fixed random cotangent.
- Order: the workload order is shuffled with a fixed seed. `confirm.py` times gather, matmul and pr4418 together in each fresh process.
- Correctness: each output (and gradient for `fwd_bwd`) is compared with `gather` in float32.
  The output limit is 1e-5 (float32) and 1e-2 (float16). For float32 rows, `grad_rel_err` must also be at most 1e-5 (a relative limit, because large scales sum many values per input pixel).
  Rows that fail are kept with `valid = false` and dropped by `analyze.py`.
  Every `fwd_bwd` row compares its gradient with one float32 reference (`gather` on a float32 copy of the input).
  It records `grad_max_diff` and `grad_rel_err` (`grad_max_diff` divided by the largest absolute reference gradient).
  The float16 gradient error is recorded, not gated. Float16 accumulation in the scatter-add depends on
  the order of the additions, so even `gather` differs from the float32 reference. We measure this error
  and report it instead of dropping the rows.
  A failing variant call is stored with an `error` text.
- Rows are appended to `results/sweep.jsonl`. A rerun skips keys that already exist.

## How to run

Run from any folder with the `cs6886-mlx-upsample` environment.

```sh
python experiments/01_crossover/sweep.py --smoke        # about 30 s, writes results/scratch/
python experiments/01_crossover/sweep.py                # full sweep, results/sweep.jsonl and meta.json
python experiments/01_crossover/analyze.py              # crossover.csv, figures/, gate_regret.csv, summary.md
python experiments/01_crossover/confirm.py              # 10 fresh processes per key, results/confirm.jsonl
python experiments/01_crossover/analyze.py --in experiments/01_crossover/results/scratch/sweep.jsonl
```

`sweep.py` options: `--set grid|real|all`, `--pass fwd|fwd_bwd|both`, `--out PATH`.
`confirm.py` uses the `rife.*` workloads and the grid sizes at the switch points in `crossover.csv`.
For each group where matmul wins, it uses the last winning size and the next size above it.
It also uses the first winning size and the size below it, unless the first winning size is the smallest one.
Groups where matmul never wins are skipped. Run `analyze.py` first. Use `--names` to choose workloads, and `--smoke` for a short test.
`analyze.py` compares `confirm.jsonl` with the sweep and flags gaps above 15%.

## Analysis rules

- All comparisons use eager (uncompiled) `gather`. `gather_compiled` compiles the forward only; in `fwd_bwd` the backward runs uncompiled.
- A method wins at a size only if its time is below 0.9 x the other time. Other sizes are ties. Identical code paths differ by 1.2% (median), 8% (90th percentile), 15% (95th percentile) and up to 59%. The 10% margin is near the 90th percentile, so a single win or tie close to the margin can be noise.
- The gate regret uses the gather and matmul rows: the time of the method the PR #4418 rule picks minus the best of the two.
  The gate counts as wrong only if its choice is more than 10% slower than the best.
- `analyze.py` stops if the input has duplicate rows for one workload, variant and pass.

## Findings

Results are in `results/`. The full run has 3216 rows. `analyze.py` drops 5 invalid rows.
All 5 are Metal out-of-memory failures at `grid.h2048.s0.25.c256` in `fwd_bwd` (a 4.3 GB input).
The 512 MB cap limits only the output. So the scale 0.25, C=256 `fwd_bwd` groups end at H=1536 because of the memory failure, not the cap.
`results/summary.md` has the tables. Figures are in `results/figures/`.

**Hardware.** One Apple M1 laptop (MacBookPro17,1, 16 GB), mlx 0.32.4. `meta.json` shows the start of the follow-up run.
The first run started on 2026-10-21 at 09:08 UTC. All numbers come from this one machine.

**Upscaling.** Matmul is more than 10% faster than eager gather at every tested size for C=32 and C=256.
This holds for both dtypes and both passes, up to the 512 MB size cap. For C=4, matmul clearly wins up to H=1024.
At H=1536, the fresh-process runs give a gather/matmul ratio from 0.83 to 1.26, so most cases are ties.
Gather wins at scales 1.5 and 2 in float32. By H=2048 (scales 1.5 and 2), matmul no longer wins.
So the upper crossover for C=4 is around H=1536. It is far above 128.
For float32 forward at H=64 to 256, the median speedup is 5.0x, and the peak is 15.9x (`grid.h64.s4.c256`).
See `results/figures/heatmap_fwd_float32.png`.

**Downscaling.** The forward pass is mixed. At scale 0.25 and C=4, matmul never wins.
At scale 0.5, matmul wins at mid sizes. Over all downscaling groups, matmul wins 37 of 140 sizes in the forward pass.
With the backward pass, matmul wins 101 of 138 sizes. The gather backward is a scatter-add.
See `results/figures/heatmap_fwd_bwd_float32.png`.

**PR #4418 gate.** The rule picks a method more than 10% slower than the best in 369 of 802 configs.
Examples of large losses (`results/gate_regret.csv` has the full ranking):

| Config | Gate picks | Time | Best |
| --- | --- | --- | --- |
| `depth_anything_l.fuse148`, B=4, float32, fwd_bwd | gather | 508 ms | matmul 109 ms |
| `grid.h512.s4.c32`, float32, fwd_bwd | gather | 735 ms | matmul 249 ms |
| `grid.h1536.s0.25.c256`, float32, fwd_bwd | gather | 4142 ms | matmul 2108 ms |

The fresh-process check for RIFE `up4` (B=8, float32, forward) gives gather about 100 ms and matmul about 17 ms.

**float16 gradients.** The relative gradient error of gather in float16 reaches 2.9% (PSPNet at scale 30 and 60).
This is about 60 times the maximum for matmul (0.05%, near the float16 rounding level).
The two methods differ most at large scale factors. In float32, both are near 1e-7 (median).
The error is recorded, not gated. See `results/grad_error.csv`.

**mx.compile.** Compiling the forward makes gather 1.5x faster (median). The backward was not compiled in this test.

**Noise.** Identical code paths differ by 1.2% (median). 92% of the grid configs are within 10%. The largest difference is 59%.
In the fresh-process check, 137 configs were compared. The winner agrees in 120. No config has opposite winners.
The other 17 are ties on one side. The ratio gather/matmul differs by 3.0% (median) and 11.9% (90th percentile).
Fresh processes include a cold start: the median absolute gather time is 8.1% higher.
One target was skipped: `grid.h2048.s0.25.c256` float32 `fwd_bwd` (out of memory in the sweep).

**Candidate rule (M1 only; verify on other chips).**
- Upscaling: use matmul, unless the input is very large (C=4 above about H=1024 to 1536).
- Downscaling, forward only: use gather at small C.
- Downscaling with gradients: matmul in most tested sizes (101 of 138). Gather wins at the largest sizes and in some small-C float32 groups. See `results/crossover.csv`.

This is a candidate. The data does not show how it performs on other chips.

**Limitations.**
- One chip.
- The DeepLabV3+, PSPNet and RIFE shapes are not verified against configs.
- Training rows use float32 and inference rows use float16, so dtype and batch size are mixed.
- Compile covers the forward pass only.
