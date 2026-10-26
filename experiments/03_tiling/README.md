# 03 Tiling: threadgroup shared memory

## Question

`fused` (from `02_fused_kernel`) reads 4 corner pixels per thread straight from GPU main memory,
with no reuse between threads. For upscaling, nearby output pixels in the same threadgroup often
read the same or overlapping input pixels. Does loading the input tile once per threadgroup, into
fast on-chip shared memory, and reusing it across the threadgroup's output pixels (`fused_tiled` in
`mlx_upsample.kernels.bilinear`) beat the plain `fused` kernel?

Downscaling has little or no such reuse (each output pixel pulls from a mostly disjoint input
region), so this experiment only times workloads with `scale > 1`. `fused_tiled` falls back to
`fused` when either scale is below 1.

## Method

- Kernel under test: `fused_tiled(x, scale_factor, align_corners, tile=TILE)` in
  `mlx_upsample.kernels.bilinear`. Each threadgroup covers a `tile = (W, H)` block of output
  pixels; the threads in that team cooperatively load the input region the block needs into
  threadgroup memory once, then every thread reads from that shared copy instead of from main
  memory.
- Reference / baseline: `fused` (cached tables, `02_fused_kernel`'s default).
- **Tile tuning (`tune.py`)**: time `gather`, `fused`, and `fused_tiled` at tile sizes (8,2), (8,4),
  (16,4), (8,8) on a small fixed workload set (grid points spanning small/large `H`, `C` in
  {4, 32, 256}, scale in {2, 4}, both dtypes, plus the two RIFE upscale shapes from issue #4435).
  `tune_summary.md` reports, per tile, the median and geometric-mean speedup of that tile over
  `fused`, and win/tie/loss counts using the same 0.9x margin as below. The winning tile becomes
  `TILE` in `bilinear.py` (owned by the kernel author, not this folder).
- **Full sweep (`sweep.py`)**: time `gather`, `fused`, `fused_tiled` (using whatever `TILE` the
  kernel module ships with) on `grid()` and `real()`, filtered to `scale > 1`, forward pass only.
  Reuses `mlx_upsample.bench.run_sweep` for the timing loop, correctness gate, interleaved timing,
  and the resumable `results/sweep.jsonl` (see `02_fused_kernel` for how that loop works).
- Timing rule: a method wins only if it is below 0.9x the other time; otherwise the pair counts as
  a tie (same 10% noise margin as `01_crossover` and `02_fused_kernel`).
- Memory: both scripts cap output size at `--max-mb` (default 256 MB, lower than 02's 512 MB) to
  keep peak memory around 2 GB on a 16 GB machine that is also running other apps — `gather`
  allocates several output-sized temporaries, so peak memory is several times the output size. Both
  scripts also call `mx.set_cache_limit(1 << 30)` so MLX returns freed buffers to the OS between
  workloads instead of holding onto them for reuse.

## How to run

Run from any folder with the `cs6886-mlx-upsample` environment.

```sh
python experiments/03_tiling/tune.py --smoke     # 2 workloads, writes results/scratch/
python experiments/03_tiling/tune.py             # full tuning set, results/tune.jsonl + tune_summary.md

python experiments/03_tiling/sweep.py --smoke    # a few rows, writes results/scratch/
python experiments/03_tiling/sweep.py            # full grid + real catalog (scale > 1), results/sweep.jsonl
python experiments/03_tiling/analyze.py          # reads results/sweep.jsonl by default
python experiments/03_tiling/analyze.py --in experiments/03_tiling/results/scratch/sweep.jsonl

python experiments/03_tiling/rowsize.py --smoke  # 2 configs, writes results/scratch/
python experiments/03_tiling/rowsize.py          # full row-size sweep, results/rowsize.jsonl
python experiments/03_tiling/rowsize.py --analyze  # re-reads results/rowsize.jsonl, rewrites summary + figure

python experiments/03_tiling/profile.py --smoke   # 1 config, 1 variant, writes results/scratch/traces/
python experiments/03_tiling/profile.py --label v2            # records traces, results/profile_v2.csv + _summary.md
python experiments/03_tiling/profile.py --label v2 --analyze  # re-reads existing traces, no new recording

python experiments/03_tiling/improve.py --smoke    # 3 workloads, writes results/scratch/
python experiments/03_tiling/improve.py            # 64 workloads, results/improve.jsonl + improve_summary.md
python experiments/03_tiling/improve.py --analyze  # re-reads results/improve.jsonl, rewrites summary only
```

`profile.py` needs Xcode with `xctrace` (Apple's GPU trace tool) installed. Each run records one
trace per (kernel, workload) and saves it under `results/scratch/traces/<label>/`. `--analyze` skips
recording and re-reads traces already on disk.

`sweep.py` writes `results/sweep.jsonl` and `results/meta.json`, and is resumable: a second run
only fills in missing (workload, variant) rows. `analyze.py` writes `results/speedup.csv`,
`results/figures/heatmap_fwd_{float32,float16}.png`, `results/figures/real_workloads.png`, and
`results/timing_summary.md` (now with a GB/s column, and an "Effective bandwidth" bucket table).

`rowsize.py` writes `results/rowsize.jsonl`, `results/rowsize_meta.json`,
`results/rowsize_summary.md`, and `results/figures/rowsize.png`. It is also resumable, and
`--analyze` only rebuilds the summary and figure from the existing JSONL, with no new runs.

## Findings

The tuning and sweep numbers below (through "Bandwidth and the row-size test") are all for the
*first* version of `fused_tiled` — the one with a divide-based cooperative load. That sweep was not
rerun after the kernel changes described in "GPU profiling and kernel changes". `improve.py` is the
check that was rerun, on the new kernel, across 64 workloads.

### Tile tuning

The tuning run used 14 workloads, 4 tile sizes, 0 invalid rows. No tile beat `fused` on any
workload. Median speedup vs `fused`: tile (8,2) 0.636x, (8,4) 0.655x, (16,4) 0.752x, (8,8)
0.771x. Tile (8,8) was the least bad, so it is the default `TILE` in `bilinear.py`.

### Timing findings

The full sweep ran 711 rows (237 workloads: 188 grid, 49 real), forward pass only, with 0
invalid rows. `fused_tiled` never beats `fused`: 0 wins, 8 ties, 229 losses. The ratio
(`fused` time / `fused_tiled` time) has a median of 0.696x, down to 0.453x at the worst case.

Every real-model workload loses, from 0.538x to 0.881x; the worst is the `rife.up2` shape at
batch 1, float16 (`fused` 0.545 ms, `fused_tiled` 1.01 ms). For the RIFE x4 shape from issue
#4435 (batch 1, float16), `fused` takes 0.848 ms and `fused_tiled` takes 1.05 ms.

The heatmaps (`results/figures/heatmap_fwd_{float32,float16}.png`) show near-ties only at the
smallest output sizes, where a fixed per-call cost of about 0.2 ms dominates both kernels. The
loss grows as output size grows, down to about 0.55x at the largest sizes. So tiling costs more
per output pixel than `fused` does; it is not just a fixed setup cost.

Three hypotheses for why tiling does not help here:

- The M1 GPU's cache may already serve repeat reads of nearby input pixels, so `fused` gets most
  of the reuse tiling is meant to add. Tested below in "Bandwidth and the row-size test" and
  supported by that test.
- Upscaling is write-heavy: at scale 4, the kernel writes 16x more bytes than it reads. Tiling
  only cuts down the reads, so it cannot help much when writes dominate. Not measured directly.
- Tiling adds its own cost: a per-element divide and modulo during the cooperative load, a
  barrier where every thread waits for the load to finish, and threadgroup memory that may limit
  how many thread teams run at once. Not measured directly.

### Bandwidth and the row-size test

`timing_summary.md` reports effective bandwidth as `min_bytes / time`, where `min_bytes` = input
read once + float32 output written once. This is a lower bound on real DRAM traffic, so it is a
lower bound on true bandwidth too. Bucketed by `min_bytes`, `fused` rises as workloads grow, up to
a median of 47.9 GB/s on the largest bucket (100-300 MB, n=43); `fused_tiled` stays lower, 29.5
GB/s on that same bucket:

| bucket | n | fused GB/s | fused_tiled GB/s |
| --- | --- | --- | --- |
| <= 10 MB | 85 | 8.69 | 6.72 |
| 10-50 MB | 77 | 32.7 | 21.8 |
| 50-100 MB | 28 | 43.4 | 27.5 |
| 100-300 MB | 43 | 47.9 | 29.5 |

**Question.** Would bigger outputs make tiling win? Theory: output size is not the deciding
factor. Neighbouring output rows read the same input rows. If one input row (`W_in * C * 4` bytes)
fits in the GPU's cache, `fused` gets that reuse for free, at no extra code cost. Only when a row
is too big for the cache must `fused` re-read input from main memory on every pass; only then
should tiling, which avoids those re-reads, have a chance to help.

**Test (`rowsize.py`).** Output is fixed at 128 MiB of float32, with `C=64` and `B=1`. Row size
(`W_in * C * 4`) is swept from 16 KiB to 16 MiB (scale 2) and 16 KiB to 4 MiB (scale 4), by making
the image wider and shorter at each step. A `copy` kernel reads the input once and writes the
output once, with no upsampling math: it is the practical bandwidth ceiling for this amount of
traffic. `fused_tiled` matched `fused` exactly at every row size (max abs diff 0).

**Results.** `copy` holds steady at about 43-49 GB/s across all row sizes: the traffic, not the
math, sets its speed. For rows up to 1 MiB, `fused` runs at 0.97-1.06x of `copy`: it is already at
the practical ceiling, so no upscale kernel could be much faster there, and tiling loses
(`fused`/`fused_tiled` time ratio 0.61-0.83x). At 4 MiB rows, `fused` drops to 29.3 GB/s (scale 2)
and 35.6 GB/s (scale 4) — 0.66x and 0.82x of `copy` — while `fused_tiled` holds steady (31.4 and
37.0 GB/s). The `fused`/`fused_tiled` time ratio is 1.07x and 1.04x: under the 0.9x margin used
throughout this experiment, that is a tie, not a tiling win. So the cache limit sits between 1 MiB
and 4 MiB rows, and past it tiling does cut the re-reads as the theory predicts — but tiling's own
overhead (see hypothesis 3) still keeps it from winning outright.

At 16 MiB rows (scale 2, only 2 input rows total), both kernels slow down (`fused` 34.0 GB/s,
`fused_tiled` 25.5 GB/s, `copy` 47.0 GB/s). With only 2 rows, the two edge rows dominate the whole
run; this case is not explained by the cache theory above.

**Real models.** The widest row in `real()` (`mlx_upsample.workloads`, `capped=False`) is
`segformer_b2.mlp64` at 192 KiB — well inside the 1 MiB mark, so real rows fit in the GPU cache
and tiling has no re-reads left to remove. The largest real workloads run close
to the bandwidth ceiling, same as in the row-size test. Small real workloads instead run into the
fixed per-call cost (the <=10 MB bucket manages only 8.69 GB/s) — and tiling does not reduce that
cost either. Practical conclusion: for the model shapes this project targets, tiling cannot pay
off on this M1, whether the limit is cache re-reads, the bandwidth ceiling, or the fixed per-call
floor.

**Caveat.** All of the above is from one machine (M1, 16 GB), one run per config, median of
repeated calls per config; the 1-4 MiB cache boundary rests on only two data points.

### GPU profiling and kernel changes

**Method.** Xcode Instruments (`xctrace`, Metal System Trace + Metal GPU Counters), one trace per
(kernel, workload). Counters are averaged over the time our compute work actually ran on the GPU.
`v1` = before the changes below, `v2` = after.

**v1 findings.** On the two row-size configs, `fused_tiled` was ALU-bound: ALU Limiter was ~88-90%
(89.95% at 1 MiB, 87.93% at 4 MiB), but F32 Utilization was only ~7%. So the math units were busy,
but not with the blend math — they were busy with integer index math. On `rife_up4`, ALU Limiter
was lower, 64.43%. The cooperative load did 4 runtime integer divides per value copied into shared
memory.

Read-bandwidth counters (bytes = GB/s x ms) showed: at both 1 MiB and 4 MiB rows, `fused` read about
3.7-3.8x the input size per call; `fused_tiled` read about 1.4x (the extra 0.4x is tile-border
overlap); `copy` read about 1x. So `fused` does not keep rows in the GPU's own cache at either row
size. At 1 MiB the repeats are still cheap — our reading of the counters (not a documented fact) is
that they come from a larger cache outside the GPU, call it the system-level cache. At 4 MiB the
repeats come from DRAM. This refines the row-size theory above: the cache that matters for `fused`
is not only the GPU's own cache.

Also, `fused` on the `rife.up4` shape had low Compute Occupancy (~35%): with `C=4`, one threadgroup
is only 32 threads.

**Changes.** (1) A division-free load loop in `bilinear_tiled.metal`, to remove the integer divides
above. (2) A `tg_rows` option on `fused` — a taller threadgroup, covering more output rows per team;
default stays 1 (unchanged behaviour).

**v2 counters.** `fused_tiled`'s ALU Limiter fell to ~62-70%. `fused_rows4`/`fused_rows8` halved
`fused`'s read traffic at both row sizes: read bandwidth went from 35.4 to 15.7/17.1 GB/s at 1 MiB,
and from 24.5 to 14.8/14.9 GB/s at 4 MiB. A taller team covers neighbouring output rows, so the
GPU's own cache now catches the repeats that `fused` (tg_rows=1) was missing. `tg_rows` did not
raise occupancy on the RIFE shapes: it stayed at 33-38%.

**Caveat.** The profiler's GPU ms/call drifts between sessions — `fused`, with unchanged code,
measured 2.93 ms in v1 and 3.59 ms in v2 at 1 MiB rows. So only compare kernels within one profiling
run. The real verdict on whether the changes help comes from `improve.py`, which times kernels
against each other within a single run.

**`improve.py` results.** Interleaved timing (same method as `tune.py`/`sweep.py`) over 64
workloads (49 real, 12 grid, 3 row-size; 0 invalid), ratio = `fused` time / variant time, 0.9x win
margin. On real workloads: `fused_rows4` wins 3, ties 45, loses 1; `fused_rows8` wins 1, ties 44,
loses 4; both have a median ratio near 0.99x — no real gain. `fused_tiled`, even with the new load
loop, still loses on real workloads: 0 wins, 11 ties, 38 losses, median 0.756x (the first kernel's
sweep median was 0.696x — a smaller loss, but still a loss).

On 4 MiB rows the new `fused_tiled` wins (`rowsize.s2.4MiB`: `fused` 6.20 ms, `fused_tiled` 4.51 ms;
`rowsize.s4.4MiB`: 4.44 ms vs 3.95 ms), but taller threadgroups match it with no shared memory and
no barrier (`fused_rows8`: 4.43 ms and 3.74 ms). At 1 MiB rows, nothing beats `fused`.

**Conclusion.** The code can be improved, but only for very wide input rows (above roughly 1-4
MiB), and no real model in the catalog has rows that wide (the widest is 192 KiB). There, the
simplest fix is a taller threadgroup, not tiling. For real workloads, `fused` (default `tg_rows=1`)
stays the default: neither change helps them. The remaining cost on real workloads is mostly the
fixed per-call cost, not the kernel.

Conclusion: keep `fused` as the default upscale kernel. `fused_tiled` stays in the code as the
recorded result of this experiment. Downscaling was not tested here, since `fused_tiled` falls
back to `fused` for scale < 1.

A first version of the tune run skipped the output-size cap and pushed the machine into swap
(moved to `results/scratch/broken_tune/`). After adding the cap, swap stayed flat at about
2.6-3.1 GB through both the tune run (42 s) and the sweep run (6.5 min).

See `results/figures/heatmap_fwd_float32.png`, `results/figures/heatmap_fwd_float16.png`, and
`results/figures/real_workloads.png` for the full picture.
