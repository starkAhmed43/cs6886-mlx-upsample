# Summary

Source: `sweep.jsonl`

- Rows: 2010 (0 invalid and dropped)
- Grid configs: 346; real configs: 56
- A method wins only if it is below 0.9 x the other time; otherwise the pair counts as a tie (same 10% noise margin as 01_crossover).
- A workload is excluded from a baseline's win/tie/loss counts (and from best_other) when that baseline has no valid row for it (e.g. invalid/OOM).

## Wins (fused vs each baseline)

- vs gather: {'fused': 401, 'tie': 1} (0 excluded, no gather data)
- vs gather_compiled: {'fused': 375, 'tie': 26, 'other': 1} (0 excluded, no gather_compiled data)
- vs matmul: {'fused': 400, 'tie': 2} (0 excluded, no matmul data)
- vs pr4418: {'fused': 400, 'tie': 2} (0 excluded, no pr4418 data)
- vs best_other: {'fused': 373, 'tie': 28, 'other': 1} (0 excluded, no baseline data at all)

## Speedup (fused vs best of the non-fused variants)

- Overall: median 2.52x, max 22.14x
- Upscale (scale >= 1): median 2.64x, max 22.14x
- Downscale (scale < 1): median 2.13x, max 8.67x
- float32: median 2.43x, max 22.14x
- float16: median 2.57x, max 21.83x

## Invalid rows

(none)

## Real workloads

| name | set | B | H | W | C | scale | align_corners | dtype | pass | fused | gather | gather_compiled | matmul | pr4418 | speedup_vs_best_other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rife.up2 | real | 8 | 384 | 640 | 4 | 2 | False | float32 | fwd | 3.8 | 98.2 | 62.3 | 34.6 | 92.7 | 9.1 |
| rife.up2 | real | 1 | 384 | 640 | 4 | 2 | False | float16 | fwd | 1.02 | 14.8 | 8.07 | 5.86 | 14.7 | 5.74 |
| depth_anything_l.fuse148 | real | 4 | 148 | 148 | 256 | 2 | True | float32 | fwd | 8.49 | 261 | 177 | 45.2 | 260 | 5.33 |
| rife.up4 | real | 8 | 192 | 320 | 4 | 4 | False | float32 | fwd | 3.44 | 93.8 | 61.8 | 16.5 | 93.3 | 4.81 |
| rife.down2 | real | 8 | 768 | 1280 | 4 | 0.5 | False | float32 | fwd | 3.51 | 23.8 | 16.2 | 32.9 | 23.9 | 4.61 |
| depth_anything_l.fuse148 | real | 1 | 148 | 148 | 256 | 2 | True | float16 | fwd | 2.64 | 74.9 | 42 | 11.6 | 74.5 | 4.39 |
| depth_anything_s.fuse148 | real | 4 | 148 | 148 | 64 | 2 | True | float32 | fwd | 2.79 | 66 | 45.2 | 11.8 | 65.9 | 4.23 |
| sam.masks | real | 4 | 256 | 256 | 3 | 4 | False | float32 | fwd | 1.9 | 38.6 | 25.2 | 7.04 | 38.2 | 3.71 |
| depth_anything_l.fuse74 | real | 4 | 74 | 74 | 256 | 2 | True | float32 | fwd | 2.75 | 65.9 | 45.3 | 9.36 | 9.42 | 3.41 |
| rife.up4 | real | 1 | 192 | 320 | 4 | 4 | False | float16 | fwd | 0.962 | 14.5 | 7.84 | 3.1 | 14.5 | 3.22 |
| upernet.logits | real | 2 | 128 | 128 | 150 | 4 | False | float32 | fwd | 6.57 | 227 | 156 | 20.7 | 20.8 | 3.16 |
| segformer.logits | real | 2 | 128 | 128 | 150 | 4 | False | float32 | fwd | 6.62 | 227 | 156 | 20.9 | 21 | 3.15 |
| sam.masks | real | 1 | 256 | 256 | 3 | 4 | False | float16 | fwd | 0.853 | 12.2 | 6.56 | 2.65 | 12.4 | 3.1 |
| deeplabv3p.logits | real | 2 | 129 | 129 | 21 | 3.98 | True | float32 | fwd | 1.78 | 34.6 | 23.2 | 5.34 | 34.7 | 3 |
| upernet.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd | 3.55 | 130 | 72.7 | 10.6 | 10.6 | 2.98 |
| segformer.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd | 3.71 | 138 | 73.8 | 10.9 | 11.1 | 2.94 |
| deeplabv3p.logits | real | 1 | 129 | 129 | 21 | 3.98 | True | float16 | fwd | 1.1 | 19.9 | 11.2 | 3.22 | 19.8 | 2.93 |
| depth_anything_s.fuse148 | real | 1 | 148 | 148 | 64 | 2 | True | float16 | fwd | 1.38 | 20.5 | 11.5 | 4.01 | 20.5 | 2.9 |
| rife.down2 | real | 1 | 768 | 1280 | 4 | 0.5 | False | float16 | fwd | 0.903 | 4.74 | 2.61 | 5.64 | 4.58 | 2.9 |
| upernet.fpn64 | real | 1 | 64 | 64 | 512 | 2 | False | float16 | fwd | 1.11 | 28.8 | 16.2 | 3.18 | 3.14 | 2.83 |
| segformer_b2.mlp64 | real | 2 | 64 | 64 | 768 | 2 | False | float32 | fwd | 2.93 | 73.7 | 50.8 | 8.21 | 8.17 | 2.79 |
| depth_anything_l.fuse74 | real | 1 | 74 | 74 | 256 | 2 | True | float16 | fwd | 1.15 | 19.8 | 11.3 | 3.18 | 3.23 | 2.78 |
| depth_anything_l.fuse37 | real | 4 | 37 | 37 | 256 | 2 | True | float32 | fwd | 1.17 | 17.4 | 12.1 | 3.09 | 3.15 | 2.63 |
| depth_anything_s.fuse74 | real | 4 | 74 | 74 | 64 | 2 | True | float32 | fwd | 1.16 | 17.4 | 11.9 | 3.01 | 3.15 | 2.61 |
| segformer_b2.mlp64 | real | 1 | 64 | 64 | 768 | 2 | False | float16 | fwd | 1.83 | 42.8 | 24.1 | 4.66 | 4.67 | 2.55 |
| upernet.fpn64 | real | 2 | 64 | 64 | 512 | 2 | False | float32 | fwd | 2.3 | 49.6 | 33.9 | 5.76 | 5.82 | 2.51 |
| depth_anything_l.fuse37 | real | 1 | 37 | 37 | 256 | 2 | True | float16 | fwd | 0.692 | 7.1 | 4 | 1.64 | 1.61 | 2.33 |
| segformer_b0.mlp64 | real | 2 | 64 | 64 | 256 | 2 | False | float32 | fwd | 1.59 | 25.7 | 17.6 | 3.76 | 3.68 | 2.32 |
| deeplabv3p.aspp | real | 2 | 33 | 33 | 256 | 3.91 | True | float32 | fwd | 1.33 | 25.9 | 17.9 | 3.11 | 3.07 | 2.32 |
| rife.down4 | real | 8 | 768 | 1280 | 4 | 0.25 | False | float32 | fwd | 2.12 | 7.13 | 4.8 | 16.9 | 7.18 | 2.27 |
| depth_anything_s.fuse37 | real | 4 | 37 | 37 | 64 | 2 | True | float32 | fwd | 0.861 | 6.73 | 4.73 | 1.93 | 1.9 | 2.2 |
| deeplabv3p.aspp | real | 1 | 33 | 33 | 256 | 3.91 | True | float16 | fwd | 0.885 | 15.6 | 8.79 | 1.93 | 2.11 | 2.18 |
| depth_anything_s.fuse74 | real | 1 | 74 | 74 | 64 | 2 | True | float16 | fwd | 0.738 | 6.67 | 4.57 | 1.58 | 1.86 | 2.14 |
| upernet.fpn32 | real | 2 | 32 | 32 | 512 | 2 | False | float32 | fwd | 1.14 | 14.7 | 9.31 | 2.4 | 2.41 | 2.11 |
| rife.down4 | real | 1 | 768 | 1280 | 4 | 0.25 | False | float16 | fwd | 0.782 | 2.3 | 1.6 | 4.07 | 2.21 | 2.05 |
| segformer_b2.mlp32 | real | 2 | 32 | 32 | 768 | 4 | False | float32 | fwd | 2.68 | 73.8 | 50.7 | 5.35 | 5.33 | 1.99 |
| segformer_b2.mlp32 | real | 1 | 32 | 32 | 768 | 4 | False | float16 | fwd | 1.69 | 43.1 | 24.3 | 3.36 | 3.36 | 1.99 |
| segformer_b0.mlp64 | real | 1 | 64 | 64 | 256 | 2 | False | float16 | fwd | 1.09 | 15.3 | 8.85 | 2.16 | 2.29 | 1.98 |
| segformer_b0.mlp32 | real | 2 | 32 | 32 | 256 | 4 | False | float32 | fwd | 1.33 | 25.9 | 17.7 | 2.54 | 2.5 | 1.88 |
| depth_anything_s.fuse37 | real | 1 | 37 | 37 | 64 | 2 | True | float16 | fwd | 0.666 | 2.76 | 1.96 | 1.25 | 1.22 | 1.83 |
| segformer_b0.mlp32 | real | 1 | 32 | 32 | 256 | 4 | False | float16 | fwd | 1.06 | 16 | 9 | 1.9 | 1.86 | 1.76 |
| upernet.fpn32 | real | 1 | 32 | 32 | 512 | 2 | False | float16 | fwd | 0.851 | 8.65 | 5.64 | 1.48 | 1.49 | 1.74 |
| upernet.fpn16 | real | 1 | 16 | 16 | 512 | 2 | False | float16 | fwd | 0.401 | 2.5 | 1.88 | 0.679 | 0.689 | 1.69 |
| segformer_b2.mlp16 | real | 1 | 16 | 16 | 768 | 8 | False | float16 | fwd | 1.56 | 42.7 | 24.1 | 2.59 | 2.59 | 1.65 |
| upernet.fpn16 | real | 2 | 16 | 16 | 512 | 2 | False | float32 | fwd | 0.612 | 4.83 | 4.11 | 1.02 | 1.01 | 1.64 |
| segformer_b2.mlp16 | real | 2 | 16 | 16 | 768 | 8 | False | float32 | fwd | 2.48 | 74 | 50.8 | 4.11 | 4.02 | 1.62 |
| segformer_b0.mlp16 | real | 2 | 16 | 16 | 256 | 8 | False | float32 | fwd | 1.35 | 25.7 | 17.8 | 2.11 | 2.08 | 1.54 |
| pspnet.bin2 | real | 1 | 2 | 2 | 512 | 30 | True | float16 | fwd | 0.634 | 7.92 | 4.72 | 0.962 | 0.896 | 1.41 |
| pspnet.bin1 | real | 2 | 1 | 1 | 512 | 60 | True | float32 | fwd | 0.951 | 12.8 | 8.9 | 1.33 | 1.33 | 1.39 |
| segformer_b0.mlp16 | real | 1 | 16 | 16 | 256 | 8 | False | float16 | fwd | 1.07 | 16 | 8.69 | 1.53 | 1.44 | 1.35 |
| pspnet.bin3 | real | 2 | 3 | 3 | 512 | 20 | True | float32 | fwd | 0.958 | 13.3 | 8.19 | 1.41 | 1.29 | 1.34 |
| pspnet.bin6 | real | 1 | 6 | 6 | 512 | 10 | True | float16 | fwd | 0.595 | 7.62 | 4.54 | 0.755 | 0.809 | 1.27 |
| pspnet.bin1 | real | 1 | 1 | 1 | 512 | 60 | True | float16 | fwd | 0.459 | 7.75 | 4.37 | 0.61 | 0.577 | 1.26 |
| pspnet.bin3 | real | 1 | 3 | 3 | 512 | 20 | True | float16 | fwd | 0.448 | 6.76 | 4 | 0.557 | 0.554 | 1.24 |
| pspnet.bin2 | real | 2 | 2 | 2 | 512 | 30 | True | float32 | fwd | 0.934 | 12 | 8.79 | 1.08 | 1.15 | 1.15 |
| pspnet.bin6 | real | 2 | 6 | 6 | 512 | 10 | True | float32 | fwd | 0.949 | 12.3 | 8.22 | 1.05 | 1.1 | 1.11 |

## Worst 10 cases for fused (vs best of the non-fused variants)

| name | set | B | H | W | C | scale | align_corners | dtype | pass | fused | best_other | speedup_vs_best_other | win_vs_best_other |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| grid.h128.s0.25.c4 | grid | 1 | 128 | 128 | 4 | 0.25 | False | float32 | fwd | 0.45 | 0.35 | 0.779 | other |
| grid.h32.s0.25.c4 | grid | 1 | 32 | 32 | 4 | 0.25 | False | float16 | fwd | 0.267 | 0.263 | 0.984 | tie |
| grid.h32.s3.c4 | grid | 1 | 32 | 32 | 4 | 3 | False | float32 | fwd | 0.504 | 0.496 | 0.984 | tie |
| grid.h64.s0.5.c4 | grid | 1 | 64 | 64 | 4 | 0.5 | False | float16 | fwd | 0.339 | 0.34 | 1 | tie |
| grid.h96.s0.5.c4 | grid | 1 | 96 | 96 | 4 | 0.5 | False | float16 | fwd | 0.544 | 0.545 | 1 | tie |
| grid.h32.s0.5.c4 | grid | 1 | 32 | 32 | 4 | 0.5 | False | float32 | fwd | 0.288 | 0.289 | 1 | tie |
| grid.h32.s0.25.c4 | grid | 1 | 32 | 32 | 4 | 0.25 | False | float32 | fwd | 0.272 | 0.274 | 1.01 | tie |
| grid.h256.s0.25.c4 | grid | 1 | 256 | 256 | 4 | 0.25 | False | float32 | fwd | 0.717 | 0.726 | 1.01 | tie |
| grid.h32.s0.25.c256 | grid | 1 | 32 | 32 | 256 | 0.25 | False | float16 | fwd | 0.66 | 0.669 | 1.01 | tie |
| grid.h256.s0.25.c4 | grid | 1 | 256 | 256 | 4 | 0.25 | False | float16 | fwd | 0.698 | 0.709 | 1.02 | tie |

