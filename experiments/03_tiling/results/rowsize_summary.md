# Row-size summary

Source: `rowsize.jsonl`

Tests whether `fused` only re-reads input from DRAM once one input row (W_in * C * 4 bytes) exceeds the GPU cache; `fused_tiled` avoids most of those re-reads, and `copy` is the practical bandwidth ceiling for this input/output traffic.


## scale=2

| row_bytes | H | W | fused ms | fused_tiled ms | copy ms | fused GB/s | fused_tiled GB/s | copy GB/s | fused/copy | fused/fused_tiled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16,384 | 2048 | 64 | 3.7231 | 5.8021 | 3.6383 | 45.1 | 28.9 | 46.1 | 0.977 | 0.642 |
| 65,536 | 512 | 256 | 3.4696 | 5.4717 | 3.4368 | 48.4 | 30.7 | 48.8 | 0.991 | 0.634 |
| 262,144 | 128 | 1024 | 3.2568 | 5.3549 | 3.4428 | 51.5 | 31.3 | 48.7 | 1.057 | 0.608 |
| 1,048,576 | 32 | 4096 | 3.7408 | 5.9536 | 3.6577 | 44.8 | 28.2 | 45.9 | 0.978 | 0.628 |
| 4,194,304 | 8 | 16384 | 5.7297 | 5.3396 | 3.7848 | 29.3 | 31.4 | 44.3 | 0.661 | 1.073 |
| 16,777,216 | 2 | 65536 | 4.9367 | 6.5769 | 3.5718 | 34.0 | 25.5 | 47.0 | 0.724 | 0.751 |

## scale=4

| row_bytes | H | W | fused ms | fused_tiled ms | copy ms | fused GB/s | fused_tiled GB/s | copy GB/s | fused/copy | fused/fused_tiled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16,384 | 512 | 64 | 3.3306 | 4.3502 | 3.2972 | 42.8 | 32.8 | 43.3 | 0.990 | 0.766 |
| 65,536 | 128 | 256 | 3.1413 | 4.2506 | 3.0753 | 45.4 | 33.5 | 46.4 | 0.979 | 0.739 |
| 262,144 | 32 | 1024 | 2.8749 | 4.0425 | 2.9233 | 49.6 | 35.3 | 48.8 | 1.017 | 0.711 |
| 1,048,576 | 8 | 4096 | 3.6884 | 4.4219 | 3.5806 | 38.7 | 32.3 | 39.8 | 0.971 | 0.834 |
| 4,194,304 | 2 | 16384 | 4.0027 | 3.8568 | 3.2956 | 35.6 | 37.0 | 43.3 | 0.823 | 1.038 |

Max abs diff, fused vs fused_tiled, across all configs: 0.00e+00

