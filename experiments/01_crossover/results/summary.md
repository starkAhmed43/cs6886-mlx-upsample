# Summary

Source: `sweep.jsonl`

- Rows: 3216 (5 invalid and dropped)
- Grid configs: 691; real configs: 112
- Gate wrong (PR #4418 would pick a method more than 10% slower than the best of gather and matmul): 369 of 802
- All comparisons use eager (uncompiled) `gather`. A method wins only if it is below 0.9 x the other time; otherwise the size counts as a tie.
- Crossover groups (pass, scale, C, dtype): {'always': 38, 'partial': 30, 'never': 4}
- Median gather / gather_compiled (grid): {'fwd': 1.5, 'fwd_bwd': 1.0}. `gather_compiled` compiles the forward only. In `fwd_bwd` the backward runs uncompiled, so this table does not show what compiling the backward would do.

## Invalid rows

| name | dtype | pass | variant | error | rows |
| --- | --- | --- | --- | --- | --- |
| grid.h2048.s0.25.c256 | float16 | fwd_bwd | matmul | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | fwd_bwd | gather | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | fwd_bwd | gather_compiled | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | fwd_bwd | matmul | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | fwd_bwd | pr4418 | OOM | 1 |

## Matmul wins over gather, by scale and C

`always` and `never` cover all sizes up to the largest size that fits the memory cap (`max_H`). Otherwise the sizes where matmul wins are listed. `(gaps)` means gather clearly wins at a size between the first and last matmul win.

| scale | C | fwd float16 | fwd float32 | fwd_bwd float16 | fwd_bwd float32 |
| --- | --- | --- | --- | --- | --- |
| 0.25 | 4 | never (to H=2048) | never (to H=2048) | H 32..256 of 2048 | H 192..1024 of 2048 |
| 0.25 | 32 | never (to H=2048) | H 128..128 of 2048 | H 32..512 of 2048 | H 64..1536 of 2048 |
| 0.25 | 256 | H 32..32 of 2048 | never (to H=2048) | H 32..512 of 1536 | always (to H=1536) |
| 0.5 | 4 | H 128..384 of 2048 | H 256..384 of 2048 | H 32..768 of 2048 | H 64..1024 of 2048 |
| 0.5 | 32 | H 32..512 of 2048 | H 64..512 of 2048 | H 32..768 of 2048 | H 32..1024 of 2048 |
| 0.5 | 256 | H 32..512 of 1024 | H 32..512 of 1024 | H 32..512 of 1024 | H 32..768 of 1024 |
| 1.5 | 4 | H 64..1024 of 2048 | H 64..1024 of 2048 | H 32..1024 of 2048 | H 32..1024 of 2048 |
| 1.5 | 32 | always (to H=1024) | always (to H=1024) | always (to H=1024) | always (to H=1024) |
| 1.5 | 256 | always (to H=384) | always (to H=384) | always (to H=384) | always (to H=384) |
| 2 | 4 | H 32..1536 of 2048 | H 32..1024 of 2048 | H 32..1024 of 2048 | H 32..1024 of 2048 |
| 2 | 32 | always (to H=1024) | always (to H=1024) | always (to H=1024) | always (to H=1024) |
| 2 | 256 | always (to H=256) | always (to H=256) | always (to H=256) | always (to H=256) |
| 3 | 4 | H 64..1536 of 1536 | always (to H=1536) | H 32..1024 of 1536 | H 32..1024 of 1536 |
| 3 | 32 | always (to H=512) | always (to H=512) | always (to H=512) | always (to H=512) |
| 3 | 256 | always (to H=192) | always (to H=192) | always (to H=192) | always (to H=192) |
| 4 | 4 | always (to H=1024) | always (to H=1024) | always (to H=1024) | always (to H=1024) |
| 4 | 32 | always (to H=512) | always (to H=512) | always (to H=512) | always (to H=512) |
| 4 | 256 | always (to H=128) | always (to H=128) | always (to H=128) | always (to H=128) |

## Gradient error against a float32 reference

Relative error is the largest absolute difference divided by the largest absolute reference gradient. All rows with a gradient are included, also rows marked invalid.

| variant | dtype | median | max |
| --- | --- | --- | --- |
| gather | float16 | 0.000895 | 0.0286 |
| gather | float32 | 1.01e-07 | 2.3e-06 |
| gather_compiled | float16 | 0.000892 | 0.0224 |
| gather_compiled | float32 | 1.02e-07 | 2.89e-06 |
| matmul | float16 | 0.000352 | 0.000485 |
| matmul | float32 | 1.42e-07 | 2.1e-06 |
| pr4418 | float16 | 0.000379 | 0.00151 |
| pr4418 | float32 | 1.27e-07 | 2.1e-06 |

Ten worst rows:

| name | B | scale | C | dtype | variant | grad_rel_err |
| --- | --- | --- | --- | --- | --- | --- |
| pspnet.bin2 | 1 | 30 | 512 | float16 | gather | 0.0286 |
| pspnet.bin1 | 1 | 60 | 512 | float16 | gather | 0.0232 |
| pspnet.bin2 | 1 | 30 | 512 | float16 | gather_compiled | 0.0224 |
| pspnet.bin1 | 1 | 60 | 512 | float16 | gather_compiled | 0.0201 |
| pspnet.bin3 | 1 | 20 | 512 | float16 | gather | 0.00593 |
| pspnet.bin3 | 1 | 20 | 512 | float16 | gather_compiled | 0.00485 |
| segformer_b0.mlp16 | 1 | 8 | 256 | float16 | gather | 0.00405 |
| segformer_b2.mlp16 | 1 | 8 | 768 | float16 | gather_compiled | 0.00342 |
| segformer_b2.mlp16 | 1 | 8 | 768 | float16 | gather | 0.00288 |
| segformer_b0.mlp16 | 1 | 8 | 256 | float16 | gather_compiled | 0.00279 |

## Noise control

pr4418 runs the same code as matmul (scale >= 1, H <= 128) or as gather (all other cases). Over 690 grid configs, the median difference is 1.2% and the 90th percentile is 8.1%, 95th percentile is 15.0% and the largest is 59.0%. The 10% margin is near the 90th percentile, so a single win or tie close to the margin can be noise.

## Top 10 gate regrets (ms lost by the PR #4418 choice against the best of gather and matmul)

| name | B | dtype | pass | gather | matmul | chosen | regret_ms | ratio | gate_wrong |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid.h1536.s0.25.c256 | 1 | float32 | fwd_bwd | 4.14e+03 | 2.11e+03 | 4.14e+03 | 2.03e+03 | 1.97 | True |
| grid.h512.s4.c32 | 1 | float16 | fwd_bwd | 907 | 258 | 907 | 649 | 3.51 | True |
| grid.h512.s4.c32 | 1 | float32 | fwd_bwd | 735 | 249 | 735 | 486 | 2.95 | True |
| grid.h192.s3.c256 | 1 | float16 | fwd_bwd | 560 | 95 | 560 | 465 | 5.9 | True |
| depth_anything_l.fuse148 | 4 | float32 | fwd_bwd | 508 | 109 | 508 | 399 | 4.66 | True |
| grid.h512.s4.c32 | 1 | float16 | fwd | 518 | 119 | 518 | 399 | 4.35 | True |
| grid.h384.s1.5.c256 | 1 | float16 | fwd_bwd | 561 | 185 | 561 | 376 | 3.03 | True |
| grid.h512.s3.c32 | 1 | float16 | fwd_bwd | 520 | 155 | 520 | 365 | 3.34 | True |
| grid.h192.s3.c256 | 1 | float32 | fwd_bwd | 444 | 83.4 | 444 | 360 | 5.32 | True |
| grid.h384.s4.c32 | 1 | float16 | fwd_bwd | 456 | 104 | 456 | 352 | 4.39 | True |

## Fresh-process check against the sweep (`confirm.jsonl`)

Fresh-process runs include a cold start, so absolute times are higher. The median absolute gather time gap is 8.1%. The check therefore compares the ratio gather / matmul and the winner (same 0.9 margin).

- Configs compared: 137
- Same winner: 120
- Opposite winners (one says matmul, the other gather): 0
- Ratio gap |log(r_fresh / r_sweep)|: median 3.0%, 90th percentile 11.9%

Configs with a different winner:

| name | B | dtype | pass | r_fresh | r_sweep | winner_fresh | winner_sweep | opposite | ratio_gap_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid.h1024.s0.5.c256 | 1 | float32 | fwd_bwd | 1.09 | 0.739 | tie | gather | False | 38.9 |
| grid.h128.s0.25.c32 | 1 | float32 | fwd | 1.02 | 1.12 | tie | matmul | False | 8.96 |
| grid.h128.s0.5.c4 | 1 | float16 | fwd | 1.09 | 1.12 | tie | matmul | False | 2.64 |
| grid.h1536.s0.25.c4 | 1 | float32 | fwd_bwd | 1.25 | 1.09 | matmul | tie | False | 13.9 |
| grid.h1536.s2.c4 | 1 | float16 | fwd | 1.09 | 1.11 | tie | matmul | False | 2.19 |
| grid.h1536.s3.c4 | 1 | float32 | fwd | 1.09 | 1.14 | tie | matmul | False | 4.47 |
| grid.h2048.s0.25.c32 | 1 | float32 | fwd_bwd | 0.948 | 0.844 | tie | gather | False | 11.6 |
| grid.h256.s0.5.c4 | 1 | float32 | fwd | 1.08 | 1.17 | tie | matmul | False | 8.47 |
| grid.h32.s0.25.c256 | 1 | float16 | fwd | 1.07 | 1.22 | tie | matmul | False | 13.5 |
| grid.h32.s1.5.c4 | 1 | float16 | fwd | 1.11 | 1.11 | matmul | tie | False | 0.597 |
| grid.h32.s3.c4 | 1 | float16 | fwd | 1.31 | 1.11 | matmul | tie | False | 16.8 |
| grid.h384.s0.5.c4 | 1 | float16 | fwd | 1.06 | 1.19 | tie | matmul | False | 12.2 |
| grid.h384.s0.5.c4 | 1 | float32 | fwd | 1.02 | 1.14 | tie | matmul | False | 10.7 |
| grid.h512.s0.5.c4 | 1 | float32 | fwd | 1.05 | 0.887 | tie | gather | False | 16.9 |
| grid.h64.s0.5.c32 | 1 | float32 | fwd | 1.07 | 1.25 | tie | matmul | False | 15.7 |
| grid.h768.s0.5.c256 | 1 | float16 | fwd_bwd | 1.19 | 1.1 | matmul | tie | False | 7.37 |
| grid.h96.s0.25.c32 | 1 | float32 | fwd | 0.999 | 0.87 | tie | gather | False | 13.8 |

## Candidate gate rule

Read the rule from the table above: use matmul for a given scale and C only for H up to the largest winning size, and only where the table shows one block of wins. Where the table shows gaps or differs between dtypes, the data does not support one simple limit.
