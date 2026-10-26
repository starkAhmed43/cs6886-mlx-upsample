# Summary

Source: `sweep.jsonl`

- Rows: 1524 (5 invalid and dropped)
- Grid configs: 327; real configs: 53
- 0 workload(s) excluded from the speedup table (missing fused row).
- `speedup_X` = X time / fused time; above 1 means fused is faster. A method wins only if it is below 0.9x the other (fused wins) or above 1/0.9x (the other wins); otherwise it is a tie (same 10% noise margin as 01_crossover/02_fused_kernel/03_tiling).

## Wins vs best_stock (min of gather, matmul, pr4418), forward+backward

- overall: {'fused': 376, 'tie': 2, 'other': 2}, speedup vs best_stock median 3.489x, min 0.593x, max 25.464x
- set=grid: {'fused': 326, 'other': 1}, speedup vs best_stock median 3.609x, min 0.593x, max 25.464x
- set=real: {'fused': 50, 'tie': 2, 'other': 1}, speedup vs best_stock median 3.244x, min 0.652x, max 10.042x
- direction=downscale: {'fused': 142, 'other': 1}, speedup vs best_stock median 3.708x, min 0.593x, max 16.459x
- direction=upscale: {'fused': 234, 'tie': 2, 'other': 1}, speedup vs best_stock median 3.341x, min 0.652x, max 25.464x
- dtype=float16: {'fused': 190, 'tie': 1, 'other': 1}, speedup vs best_stock median 3.670x, min 0.652x, max 25.464x
- dtype=float32: {'fused': 186, 'tie': 1, 'other': 1}, speedup vs best_stock median 3.304x, min 0.593x, max 21.705x

## Invalid rows

| name | dtype | variant | error | rows |
| --- | --- | --- | --- | --- |
| grid.h2048.s0.25.c256 | float16 | matmul | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | fused | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | gather | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | matmul | OOM | 1 |
| grid.h2048.s0.25.c256 | float32 | pr4418 | OOM | 1 |

## Gradient error against a float32 gather reference

Relative error is the largest absolute difference divided by the largest absolute reference gradient. float32 is gated at <= 1e-5 (invalid otherwise); float16 is recorded, not gated.

| variant | dtype | median | max | n | n_invalid |
| --- | --- | --- | --- | --- | --- |
| fused | float16 | 0.000352 | 0.000485 | 192 | 0 |
| fused | float32 | 1.79e-07 | 2.07e-06 | 188 | 0 |
| gather | float16 | 0.000885 | 0.0241 | 192 | 0 |
| gather | float32 | 9.93e-08 | 1.26e-06 | 188 | 0 |
| matmul | float16 | 0.000352 | 0.000485 | 191 | 0 |
| matmul | float32 | 1.39e-07 | 2.05e-06 | 188 | 0 |
| pr4418 | float16 | 0.000375 | 0.00156 | 192 | 0 |
| pr4418 | float32 | 1.23e-07 | 2.05e-06 | 188 | 0 |

Ten worst rows:

| name | B | scale | C | dtype | variant | grad_rel_err |
| --- | --- | --- | --- | --- | --- | --- |
| pspnet.bin2 | 1 | 30 | 512 | float16 | gather | 0.0241 |
| pspnet.bin1 | 1 | 60 | 512 | float16 | gather | 0.0153 |
| pspnet.bin3 | 1 | 20 | 512 | float16 | gather | 0.00506 |
| segformer_b0.mlp16 | 1 | 8 | 256 | float16 | gather | 0.00405 |
| segformer_b2.mlp16 | 1 | 8 | 768 | float16 | gather | 0.00322 |
| pspnet.bin6 | 1 | 10 | 512 | float16 | gather | 0.00241 |
| grid.h32.s4.c256 | 1 | 4 | 256 | float16 | gather | 0.00216 |
| segformer_b0.mlp32 | 1 | 4 | 256 | float16 | gather | 0.00216 |
| upernet.logits | 1 | 4 | 150 | float16 | gather | 0.0016 |
| grid.h128.s4.c256 | 1 | 4 | 256 | float16 | gather | 0.00159 |

## Real workloads

| name | set | B | H | W | C | scale | align_corners | dtype | pass | fused | gather | matmul | pr4418 | best_stock | speedup_best_stock | win_best_stock |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rife.down2 | real | 8 | 768 | 1280 | 4 | 0.5 | False | float32 | fwd_bwd | 6 | 73.6 | 60.3 | 70.4 | 60.3 | 10 | fused |
| rife.down2 | real | 1 | 768 | 1280 | 4 | 0.5 | False | float16 | fwd_bwd | 0.973 | 10.4 | 9.75 | 10.5 | 9.75 | 10 | fused |
| rife.up2 | real | 8 | 384 | 640 | 4 | 2 | False | float32 | fwd_bwd | 7.73 | 177 | 72.4 | 178 | 72.4 | 9.36 | fused |
| rife.up2 | real | 1 | 384 | 640 | 4 | 2 | False | float16 | fwd_bwd | 1.39 | 27.4 | 10.5 | 27.3 | 10.5 | 7.55 | fused |
| rife.down4 | real | 8 | 768 | 1280 | 4 | 0.25 | False | float32 | fwd_bwd | 4.15 | 39.8 | 26.3 | 39.8 | 26.3 | 6.35 | fused |
| rife.down4 | real | 1 | 768 | 1280 | 4 | 0.25 | False | float16 | fwd_bwd | 0.861 | 5.3 | 5.7 | 5.33 | 5.3 | 6.15 | fused |
| depth_anything_l.fuse148 | real | 1 | 148 | 148 | 256 | 2 | True | float16 | fwd_bwd | 4.44 | 141 | 26.8 | 142 | 26.8 | 6.03 | fused |
| depth_anything_s.fuse148 | real | 4 | 148 | 148 | 64 | 2 | True | float32 | fwd_bwd | 4.59 | 128 | 24.3 | 127 | 24.3 | 5.29 | fused |
| depth_anything_s.fuse148 | real | 1 | 148 | 148 | 64 | 2 | True | float16 | fwd_bwd | 1.55 | 38.2 | 7.44 | 38 | 7.44 | 4.81 | fused |
| deeplabv3p.logits | real | 2 | 129 | 129 | 21 | 3.98 | True | float32 | fwd_bwd | 2.3 | 60.5 | 10.1 | 60.3 | 10.1 | 4.42 | fused |
| depth_anything_l.fuse74 | real | 1 | 74 | 74 | 256 | 2 | True | float16 | fwd_bwd | 1.44 | 35.4 | 6.3 | 6.24 | 6.24 | 4.34 | fused |
| depth_anything_l.fuse74 | real | 4 | 74 | 74 | 256 | 2 | True | float32 | fwd_bwd | 6.15 | 159 | 25.6 | 26.4 | 25.6 | 4.17 | fused |
| deeplabv3p.logits | real | 1 | 129 | 129 | 21 | 3.98 | True | float16 | fwd_bwd | 1.37 | 36.2 | 5.69 | 36.1 | 5.69 | 4.16 | fused |
| segformer_b2.mlp64 | real | 1 | 64 | 64 | 768 | 2 | False | float16 | fwd_bwd | 2.58 | 77.2 | 10.4 | 10.5 | 10.4 | 4.04 | fused |
| upernet.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd_bwd | 6.42 | 242 | 25.5 | 25.7 | 25.5 | 3.98 | fused |
| segformer.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd_bwd | 6.53 | 242 | 26.8 | 25.4 | 25.4 | 3.9 | fused |
| segformer_b2.mlp64 | real | 2 | 64 | 64 | 768 | 2 | False | float32 | fwd_bwd | 5.1 | 140 | 19.6 | 19.9 | 19.6 | 3.85 | fused |
| upernet.fpn64 | real | 1 | 64 | 64 | 512 | 2 | False | float16 | fwd_bwd | 1.91 | 52 | 7.25 | 7.31 | 7.25 | 3.8 | fused |
| depth_anything_s.fuse74 | real | 4 | 74 | 74 | 64 | 2 | True | float32 | fwd_bwd | 1.64 | 32.9 | 6.16 | 6.25 | 6.16 | 3.75 | fused |
| upernet.fpn64 | real | 2 | 64 | 64 | 512 | 2 | False | float32 | fwd_bwd | 3.63 | 94 | 13.1 | 13.1 | 13.1 | 3.61 | fused |
| depth_anything_l.fuse37 | real | 4 | 37 | 37 | 256 | 2 | True | float32 | fwd_bwd | 1.54 | 33.7 | 5.37 | 5.47 | 5.37 | 3.47 | fused |
| segformer_b0.mlp64 | real | 2 | 64 | 64 | 256 | 2 | False | float32 | fwd_bwd | 1.98 | 47.2 | 6.71 | 6.74 | 6.71 | 3.39 | fused |
| segformer_b0.mlp64 | real | 1 | 64 | 64 | 256 | 2 | False | float16 | fwd_bwd | 1.24 | 27.1 | 4.14 | 4.17 | 4.14 | 3.34 | fused |
| deeplabv3p.aspp | real | 2 | 33 | 33 | 256 | 3.91 | True | float32 | fwd_bwd | 1.66 | 46.6 | 5.94 | 5.49 | 5.49 | 3.31 | fused |
| rife.up4 | real | 8 | 192 | 320 | 4 | 4 | False | float32 | fwd_bwd | 10.2 | 171 | 33.7 | 170 | 33.7 | 3.29 | fused |
| depth_anything_s.fuse74 | real | 1 | 74 | 74 | 64 | 2 | True | float16 | fwd_bwd | 0.732 | 10.9 | 2.7 | 2.39 | 2.39 | 3.27 | fused |
| depth_anything_l.fuse37 | real | 1 | 37 | 37 | 256 | 2 | True | float16 | fwd_bwd | 0.487 | 9.6 | 1.58 | 2.07 | 1.58 | 3.24 | fused |
| rife.up4 | real | 1 | 192 | 320 | 4 | 4 | False | float16 | fwd_bwd | 1.77 | 25.2 | 5.64 | 25.2 | 5.64 | 3.19 | fused |
| upernet.fpn32 | real | 2 | 32 | 32 | 512 | 2 | False | float32 | fwd_bwd | 1.26 | 24.4 | 3.94 | 3.93 | 3.93 | 3.12 | fused |
| upernet.fpn32 | real | 1 | 32 | 32 | 512 | 2 | False | float16 | fwd_bwd | 0.609 | 13.7 | 1.87 | 1.83 | 1.83 | 3 | fused |
| upernet.fpn16 | real | 2 | 16 | 16 | 512 | 2 | False | float32 | fwd_bwd | 0.759 | 8.39 | 2.25 | 2.24 | 2.24 | 2.96 | fused |
| depth_anything_s.fuse37 | real | 4 | 37 | 37 | 64 | 2 | True | float32 | fwd_bwd | 0.796 | 9.78 | 2.35 | 2.29 | 2.29 | 2.88 | fused |
| segformer_b2.mlp32 | real | 2 | 32 | 32 | 768 | 4 | False | float32 | fwd_bwd | 4.96 | 146 | 14.2 | 14.7 | 14.2 | 2.87 | fused |
| deeplabv3p.aspp | real | 1 | 33 | 33 | 256 | 3.91 | True | float16 | fwd_bwd | 1.1 | 27.1 | 3.3 | 3.14 | 3.14 | 2.87 | fused |
| segformer_b2.mlp32 | real | 1 | 32 | 32 | 768 | 4 | False | float16 | fwd_bwd | 2.75 | 84.1 | 7.86 | 8.03 | 7.86 | 2.86 | fused |
| sam.masks | real | 1 | 256 | 256 | 3 | 4 | False | float16 | fwd_bwd | 1.68 | 21.1 | 4.63 | 20.9 | 4.63 | 2.76 | fused |
| segformer_b2.mlp16 | real | 1 | 16 | 16 | 768 | 8 | False | float16 | fwd_bwd | 2.15 | 79.2 | 5.88 | 5.95 | 5.88 | 2.74 | fused |
| segformer_b0.mlp32 | real | 1 | 32 | 32 | 256 | 4 | False | float16 | fwd_bwd | 1.24 | 28.8 | 3.4 | 3.37 | 3.37 | 2.71 | fused |
| sam.masks | real | 4 | 256 | 256 | 3 | 4 | False | float32 | fwd_bwd | 4.77 | 67.6 | 12.6 | 67.4 | 12.6 | 2.65 | fused |
| segformer_b0.mlp32 | real | 2 | 32 | 32 | 256 | 4 | False | float32 | fwd_bwd | 1.68 | 47.3 | 4.44 | 4.77 | 4.44 | 2.64 | fused |
| segformer_b2.mlp16 | real | 2 | 16 | 16 | 768 | 8 | False | float32 | fwd_bwd | 4.05 | 137 | 10.6 | 10.5 | 10.5 | 2.59 | fused |
| depth_anything_s.fuse37 | real | 1 | 37 | 37 | 64 | 2 | True | float16 | fwd_bwd | 0.536 | 4.45 | 1.38 | 1.46 | 1.38 | 2.57 | fused |
| segformer_b0.mlp16 | real | 2 | 16 | 16 | 256 | 8 | False | float32 | fwd_bwd | 1.78 | 45.3 | 4.41 | 4.33 | 4.33 | 2.44 | fused |
| upernet.fpn16 | real | 1 | 16 | 16 | 512 | 2 | False | float16 | fwd_bwd | 0.353 | 3.91 | 0.838 | 0.829 | 0.829 | 2.35 | fused |
| segformer_b0.mlp16 | real | 1 | 16 | 16 | 256 | 8 | False | float16 | fwd_bwd | 1.22 | 28.7 | 2.82 | 2.98 | 2.82 | 2.32 | fused |
| pspnet.bin6 | real | 2 | 6 | 6 | 512 | 10 | True | float32 | fwd_bwd | 1.53 | 22.5 | 2.9 | 2.8 | 2.8 | 1.83 | fused |
| pspnet.bin3 | real | 2 | 3 | 3 | 512 | 20 | True | float32 | fwd_bwd | 1.44 | 21.7 | 2.43 | 2.53 | 2.43 | 1.7 | fused |
| pspnet.bin3 | real | 1 | 3 | 3 | 512 | 20 | True | float16 | fwd_bwd | 1.23 | 17.5 | 1.92 | 1.94 | 1.92 | 1.56 | fused |
| pspnet.bin6 | real | 1 | 6 | 6 | 512 | 10 | True | float16 | fwd_bwd | 0.948 | 15.6 | 1.64 | 1.32 | 1.32 | 1.39 | fused |
| pspnet.bin2 | real | 2 | 2 | 2 | 512 | 30 | True | float32 | fwd_bwd | 1.64 | 21 | 2.33 | 2.07 | 2.07 | 1.26 | fused |
| pspnet.bin1 | real | 2 | 1 | 1 | 512 | 60 | True | float32 | fwd_bwd | 1.97 | 21.3 | 2.23 | 2.13 | 2.13 | 1.08 | tie |
| pspnet.bin2 | real | 1 | 2 | 2 | 512 | 30 | True | float16 | fwd_bwd | 2.12 | 23.8 | 2.05 | 2.14 | 2.05 | 0.966 | tie |
| pspnet.bin1 | real | 1 | 1 | 1 | 512 | 60 | True | float16 | fwd_bwd | 1.53 | 14.2 | 0.994 | 1.03 | 0.994 | 0.652 | other |

