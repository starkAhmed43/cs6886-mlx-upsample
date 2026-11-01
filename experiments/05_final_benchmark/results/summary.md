# Summary

Source: `runs.jsonl`

- Rows: 3180 (0 invalid and dropped)
- Workloads: 28 (of 28 with any valid rows)
- Each (workload, variant, pass) median is over up to 10 fresh-process runs; `spread` in `table.csv` is (max - min) / median across those runs.
- `speedup_vs_X` = X's median ms / fused_cpp's median ms; above 1 means fused_cpp is faster. A method wins only if it is below 0.9x the other (fused_cpp wins) or above 1/0.9x (the other wins); otherwise it is a tie (same 10% noise margin as 01_crossover/02_fused_kernel/03_tiling/04_backward).

## Geometric mean speedup and win/tie/loss, fused_cpp vs stock nn.Upsample path

### fwd

- vs stock: geomean speedup 16.633x, win/tie/loss {'fused_cpp': 53}
- vs pr4418: geomean speedup 5.304x, win/tie/loss {'fused_cpp': 53}

### fwd_bwd

- vs stock: geomean speedup 13.927x, win/tie/loss {'fused_cpp': 53}
- vs pr4418: geomean speedup 4.692x, win/tie/loss {'fused_cpp': 51, 'tie': 2}

## Output dtype by variant and input dtype

| variant | dtype | out_dtype |
| --- | --- | --- |
| fused_cpp | float16 | ['mlx.core.float16'] |
| fused_cpp | float32 | ['mlx.core.float32'] |
| pr4418 | float16 | ['mlx.core.float32'] |
| pr4418 | float32 | ['mlx.core.float32'] |
| stock | float16 | ['mlx.core.float32'] |
| stock | float32 | ['mlx.core.float32'] |

## Max gradient relative error by variant and input dtype (fwd_bwd pass)

| variant | dtype | grad_rel_err_max |
| --- | --- | --- |
| fused_cpp | float16 | 0.000616 |
| fused_cpp | float32 | 2.3e-06 |
| pr4418 | float16 | 0.00157 |
| pr4418 | float32 | 2.56e-06 |
| stock | float16 | 0.027 |
| stock | float32 | 1.84e-06 |

## Invalid rows

(none)

## Fastest and slowest real workloads (by speedup vs stock)

| name | set | B | H | W | C | scale | align_corners | dtype | pass | stock | pr4418 | fused_cpp | speedup_vs_stock | speedup_vs_pr4418 | win_vs_stock | win_vs_pr4418 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| depth_anything_l.fuse148 | real | 1 | 148 | 148 | 256 | 2 | True | float16 | fwd | 75.6 | 75.8 | 1.6 | 47.1 | 47.2 | fused_cpp | fused_cpp |
| depth_anything_l.fuse148 | real | 1 | 148 | 148 | 256 | 2 | True | float16 | fwd_bwd | 137 | 137 | 3.7 | 37.1 | 37 | fused_cpp | fused_cpp |
| depth_anything_s.fuse148 | real | 1 | 148 | 148 | 64 | 2 | True | float16 | fwd | 20.1 | 20.1 | 0.552 | 36.4 | 36.4 | fused_cpp | fused_cpp |
| depth_anything_l.fuse74 | real | 1 | 74 | 74 | 256 | 2 | True | float16 | fwd | 20.1 | 3.12 | 0.554 | 36.2 | 5.63 | fused_cpp | fused_cpp |
| deeplabv3p.logits | real | 1 | 129 | 129 | 21 | 3.98 | True | float16 | fwd | 20 | 20 | 0.572 | 35 | 34.9 | fused_cpp | fused_cpp |
| upernet.fpn16 | real | 2 | 16 | 16 | 512 | 2 | False | float32 | fwd | 2.27 | 0.85 | 0.318 | 7.13 | 2.68 | fused_cpp | fused_cpp |
| rife.down4 | real | 1 | 768 | 1280 | 4 | 0.25 | False | float16 | fwd_bwd | 5.26 | 5.19 | 0.74 | 7.1 | 7 | fused_cpp | fused_cpp |
| rife.up4 | real | 1 | 192 | 320 | 4 | 4 | False | float16 | fwd_bwd | 11.6 | 25.2 | 1.82 | 6.36 | 13.8 | fused_cpp | fused_cpp |
| sam.masks | real | 1 | 256 | 256 | 3 | 4 | False | float16 | fwd_bwd | 9.53 | 20.4 | 1.69 | 5.63 | 12.1 | fused_cpp | fused_cpp |
| upernet.fpn16 | real | 1 | 16 | 16 | 512 | 2 | False | float16 | fwd | 1.33 | 0.675 | 0.293 | 4.54 | 2.3 | fused_cpp | fused_cpp |

