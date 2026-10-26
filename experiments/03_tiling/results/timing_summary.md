# Summary

Source: `sweep.jsonl`

- Rows: 711 (0 invalid and dropped)
- Grid configs: 188; real configs: 49
- 0 workload(s) excluded from the speedup table (missing fused or fused_tiled row).
- A method wins only if it is below 0.9x the other time; otherwise the pair counts as a tie (same 10% noise margin as 02_fused_kernel). ratio = fused_ms / fused_tiled_ms.

## Wins (fused_tiled vs fused)

{'fused': 229, 'tie': 8}

## Ratio (fused / fused_tiled; above 1 means tiling is faster)

- Overall: median 0.696x, min 0.453x, max 0.991x
- float32: median 0.706x, min 0.535x, max 0.990x
- float16: median 0.688x, min 0.453x, max 0.991x
- C=3: median 0.779x, min 0.763x, max 0.795x
- C=4: median 0.715x, min 0.453x, max 0.991x
- C=21: median 0.717x, min 0.698x, max 0.737x
- C=32: median 0.695x, min 0.498x, max 0.888x
- C=64: median 0.635x, min 0.552x, max 0.873x
- C=150: median 0.679x, min 0.674x, max 0.683x
- C=256: median 0.664x, min 0.524x, max 0.830x
- C=512: median 0.783x, min 0.640x, max 0.881x
- C=768: median 0.715x, min 0.624x, max 0.873x

## Effective bandwidth (GB/s), by min_bytes bucket

min_bytes = input bytes (input dtype) + output bytes (output is always float32); it counts each input byte once, so it is a lower bound on real DRAM traffic.

| bucket | n | gbs_fused | gbs_fused_tiled | ratio |
| --- | --- | --- | --- | --- |
| <= 10 MB | 85 | 8.69 | 6.72 | 1.29 |
| 10-50 MB | 77 | 32.7 | 21.8 | 1.5 |
| 50-100 MB | 28 | 43.4 | 27.5 | 1.58 |
| 100-300 MB | 43 | 47.9 | 29.5 | 1.62 |


## Invalid rows

(none)

## Real workloads

| name | set | B | H | W | C | scale | align_corners | dtype | pass | fused | fused_tiled | ratio | win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pspnet.bin2 | real | 1 | 2 | 2 | 512 | 30 | True | float16 | fwd | 0.388 | 0.44 | 0.881 | fused |
| segformer_b2.mlp32 | real | 2 | 32 | 32 | 768 | 4 | False | float32 | fwd | 2.79 | 3.19 | 0.873 | fused |
| depth_anything_s.fuse37 | real | 1 | 37 | 37 | 64 | 2 | True | float16 | fwd | 0.296 | 0.339 | 0.873 | fused |
| pspnet.bin1 | real | 2 | 1 | 1 | 512 | 60 | True | float32 | fwd | 0.526 | 0.624 | 0.843 | fused |
| segformer_b0.mlp16 | real | 1 | 16 | 16 | 256 | 8 | False | float16 | fwd | 0.543 | 0.654 | 0.83 | fused |
| pspnet.bin3 | real | 1 | 3 | 3 | 512 | 20 | True | float16 | fwd | 0.365 | 0.449 | 0.814 | fused |
| upernet.fpn16 | real | 1 | 16 | 16 | 512 | 2 | False | float16 | fwd | 0.305 | 0.377 | 0.81 | fused |
| upernet.fpn16 | real | 2 | 16 | 16 | 512 | 2 | False | float32 | fwd | 0.335 | 0.416 | 0.806 | fused |
| rife.up4 | real | 1 | 192 | 320 | 4 | 4 | False | float16 | fwd | 0.848 | 1.05 | 0.806 | fused |
| pspnet.bin3 | real | 2 | 3 | 3 | 512 | 20 | True | float32 | fwd | 0.509 | 0.634 | 0.804 | fused |
| sam.masks | real | 4 | 256 | 256 | 3 | 4 | False | float32 | fwd | 1.45 | 1.82 | 0.795 | fused |
| segformer_b0.mlp32 | real | 1 | 32 | 32 | 256 | 4 | False | float16 | fwd | 0.61 | 0.769 | 0.793 | fused |
| pspnet.bin6 | real | 1 | 6 | 6 | 512 | 10 | True | float16 | fwd | 0.404 | 0.514 | 0.787 | fused |
| pspnet.bin2 | real | 2 | 2 | 2 | 512 | 30 | True | float32 | fwd | 0.521 | 0.668 | 0.779 | fused |
| upernet.fpn32 | real | 2 | 32 | 32 | 512 | 2 | False | float32 | fwd | 0.687 | 0.887 | 0.774 | fused |
| pspnet.bin6 | real | 2 | 6 | 6 | 512 | 10 | True | float32 | fwd | 0.515 | 0.665 | 0.774 | fused |
| sam.masks | real | 1 | 256 | 256 | 3 | 4 | False | float16 | fwd | 0.503 | 0.66 | 0.763 | fused |
| segformer_b2.mlp16 | real | 2 | 16 | 16 | 768 | 8 | False | float32 | fwd | 2.21 | 2.92 | 0.757 | fused |
| segformer_b0.mlp16 | real | 2 | 16 | 16 | 256 | 8 | False | float32 | fwd | 0.951 | 1.28 | 0.741 | fused |
| segformer_b2.mlp16 | real | 1 | 16 | 16 | 768 | 8 | False | float16 | fwd | 1.18 | 1.59 | 0.74 | fused |
| deeplabv3p.logits | real | 2 | 129 | 129 | 21 | 3.98 | True | float32 | fwd | 1.28 | 1.74 | 0.737 | fused |
| segformer_b0.mlp32 | real | 2 | 32 | 32 | 256 | 4 | False | float32 | fwd | 0.888 | 1.23 | 0.722 | fused |
| deeplabv3p.logits | real | 1 | 129 | 129 | 21 | 3.98 | True | float16 | fwd | 0.682 | 0.977 | 0.698 | fused |
| pspnet.bin1 | real | 1 | 1 | 1 | 512 | 60 | True | float16 | fwd | 0.356 | 0.511 | 0.696 | fused |
| deeplabv3p.aspp | real | 1 | 33 | 33 | 256 | 3.91 | True | float16 | fwd | 0.561 | 0.806 | 0.696 | fused |
| segformer_b0.mlp64 | real | 2 | 64 | 64 | 256 | 2 | False | float32 | fwd | 1.08 | 1.55 | 0.695 | fused |
| segformer_b2.mlp32 | real | 1 | 32 | 32 | 768 | 4 | False | float16 | fwd | 1.14 | 1.66 | 0.689 | fused |
| segformer.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd | 3.17 | 4.64 | 0.683 | fused |
| upernet.logits | real | 1 | 128 | 128 | 150 | 4 | False | float16 | fwd | 3.23 | 4.78 | 0.674 | fused |
| upernet.fpn64 | real | 1 | 64 | 64 | 512 | 2 | False | float16 | fwd | 1.2 | 1.79 | 0.666 | fused |
| upernet.fpn32 | real | 1 | 32 | 32 | 512 | 2 | False | float16 | fwd | 0.39 | 0.587 | 0.664 | fused |
| deeplabv3p.aspp | real | 2 | 33 | 33 | 256 | 3.91 | True | float32 | fwd | 0.953 | 1.44 | 0.661 | fused |
| depth_anything_s.fuse37 | real | 4 | 37 | 37 | 64 | 2 | True | float32 | fwd | 0.332 | 0.504 | 0.659 | fused |
| segformer_b2.mlp64 | real | 1 | 64 | 64 | 768 | 2 | False | float16 | fwd | 1.48 | 2.27 | 0.65 | fused |
| upernet.fpn64 | real | 2 | 64 | 64 | 512 | 2 | False | float32 | fwd | 1.9 | 2.96 | 0.64 | fused |
| depth_anything_l.fuse148 | real | 1 | 148 | 148 | 256 | 2 | True | float16 | fwd | 2.63 | 4.12 | 0.638 | fused |
| depth_anything_s.fuse74 | real | 1 | 74 | 74 | 64 | 2 | True | float16 | fwd | 0.304 | 0.478 | 0.636 | fused |
| depth_anything_s.fuse74 | real | 4 | 74 | 74 | 64 | 2 | True | float32 | fwd | 0.875 | 1.38 | 0.634 | fused |
| depth_anything_l.fuse37 | real | 4 | 37 | 37 | 256 | 2 | True | float32 | fwd | 0.84 | 1.33 | 0.634 | fused |
| rife.up4 | real | 8 | 192 | 320 | 4 | 4 | False | float32 | fwd | 2.71 | 4.34 | 0.624 | fused |
| segformer_b2.mlp64 | real | 2 | 64 | 64 | 768 | 2 | False | float32 | fwd | 2.59 | 4.15 | 0.624 | fused |
| segformer_b0.mlp64 | real | 1 | 64 | 64 | 256 | 2 | False | float16 | fwd | 0.567 | 0.919 | 0.617 | fused |
| depth_anything_s.fuse148 | real | 4 | 148 | 148 | 64 | 2 | True | float32 | fwd | 2.58 | 4.23 | 0.61 | fused |
| depth_anything_l.fuse37 | real | 1 | 37 | 37 | 256 | 2 | True | float16 | fwd | 0.318 | 0.525 | 0.607 | fused |
| depth_anything_l.fuse74 | real | 1 | 74 | 74 | 256 | 2 | True | float16 | fwd | 0.714 | 1.25 | 0.572 | fused |
| depth_anything_l.fuse74 | real | 4 | 74 | 74 | 256 | 2 | True | float32 | fwd | 2.35 | 4.15 | 0.566 | fused |
| rife.up2 | real | 8 | 384 | 640 | 4 | 2 | False | float32 | fwd | 3.33 | 5.97 | 0.558 | fused |
| depth_anything_s.fuse148 | real | 1 | 148 | 148 | 64 | 2 | True | float16 | fwd | 0.667 | 1.21 | 0.552 | fused |
| rife.up2 | real | 1 | 384 | 640 | 4 | 2 | False | float16 | fwd | 0.545 | 1.01 | 0.538 | fused |

