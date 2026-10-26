# Tile tuning summary

Source: `tune.jsonl`

Workloads: 14


| tile | median speedup vs fused | geomean speedup vs fused | wins | ties | losses |
| --- | --- | --- | --- | --- | --- |
| tiled_8x2 | 0.636x | 0.624x | 0 | 1 | 13 |
| tiled_8x4 | 0.655x | 0.680x | 0 | 1 | 13 |
| tiled_16x4 | 0.752x | 0.717x | 0 | 1 | 13 |
| tiled_8x8 | 0.771x | 0.740x | 0 | 3 | 11 |
