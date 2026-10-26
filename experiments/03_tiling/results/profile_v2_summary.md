# GPU hardware counter profile

Read with Apple's `xctrace` Metal GPU Counters instrument, one trace per (config, variant). Each number is the average, over that trace's GPU counter samples, restricted to the time this process's GPU "Compute" work was running (not idle time, and not other apps' GPU work).

What each counter means:

- **GPU Read Bandwidth**: Bytes per second the GPU reads from main memory (DRAM). Unit looks like GB/s (see note below).
- **GPU Write Bandwidth**: Bytes per second the GPU writes to main memory (DRAM). Unit looks like GB/s (see note below).
- **ALU Utilization**: How busy the math (arithmetic) units are, percent of peak.
- **ALU Limiter**: How often math-unit work is stalled waiting to run, percent of peak; high means math is the bottleneck.
- **Compute Occupancy**: Percent of the GPU's compute slots that are busy running our kernel at once.
- **Buffer Read Limiter**: How often a buffer (array) read is stalled waiting to run, percent of peak; high means reads are the bottleneck.
- **Buffer Load Utilization**: How busy the buffer-read hardware is, percent of peak.
- **Buffer Write Limiter**: How often a buffer (array) write is stalled waiting to run, percent of peak; high means writes are the bottleneck.
- **Buffer Store Utilization**: How busy the buffer-write hardware is, percent of peak.
- **GPU Last Level Cache Utilization**: How busy the GPU's shared (last-level) cache is, percent of peak.
- **GPU Last Level Cache Limiter**: How often the shared cache is the reason work is stalled, percent of peak; high means the cache is the bottleneck.
- **MMU Limiter**: How often address translation on a cache miss stalls work, percent of peak; high points at DRAM, not cache.
- **Threadgroup/Imageblock Load Limiter**: How often reading fast on-chip shared memory stalls work, percent of peak.
- **Threadgroup/Imageblock Load Utilization**: How busy the fast on-chip shared-memory read path is, percent of peak.
- **Threadgroup/Imageblock Store Limiter**: How often writing fast on-chip shared memory stalls work, percent of peak.
- **Threadgroup/Imageblock Store Utilization**: How busy the fast on-chip shared-memory write path is, percent of peak.
- **F32 Utilization**: How busy the 32-bit float math units are, percent of peak.
- **F16 Utilization**: How busy the 16-bit float math units are, percent of peak (relevant for the fp16 rife.up4 config).

Note on units: `GPU Read/Write Bandwidth` values below land in the same ~20-65 range as the GB/s this project measures by wall-clock timing elsewhere (see `rowsize_summary.md`), so the counter's unit is GB/s, not a raw byte count or a percentage.


## rowsize_1MiB

| counter | fused | fused_tiled | fused_rows4 | fused_rows8 | copy |
| --- | --- | --- | --- | --- | --- |
| GPU Read Bandwidth | 35.45 | 11.75 | 15.72 | 17.05 | 9.62 |
| GPU Write Bandwidth | 34.59 | 31.19 | 31.22 | 33.93 | 36.72 |
| ALU Utilization | 31.30 | 52.49 | 28.33 | 30.40 | 7.46 |
| ALU Limiter | 37.12 | 69.75 | 34.25 | 36.85 | 8.47 |
| Compute Occupancy | 76.32 | 81.85 | 75.89 | 76.62 | 70.79 |
| Buffer Read Limiter | 18.43 | 12.83 | 16.66 | 18.00 | 1.71 |
| Buffer Load Utilization | 18.11 | 12.55 | 16.34 | 17.76 | 1.39 |
| Buffer Write Limiter | 5.53 | 4.99 | 5.11 | 5.40 | 5.94 |
| Buffer Store Utilization | 5.20 | 4.68 | 4.69 | 5.10 | 5.53 |
| GPU Last Level Cache Utilization | 14.56 | 7.88 | 8.76 | 9.49 | 7.27 |
| GPU Last Level Cache Limiter | 32.55 | 8.64 | 25.09 | 27.35 | 34.48 |
| MMU Limiter | 44.08 | 8.48 | 40.76 | 43.69 | 40.28 |
| Threadgroup/Imageblock Load Limiter | 0.22 | 5.60 | 0.20 | 0.20 | 0.21 |
| Threadgroup/Imageblock Load Utilization | 0.21 | 4.85 | 0.19 | 0.19 | 0.20 |
| Threadgroup/Imageblock Store Limiter | 0.22 | 5.60 | 0.20 | 0.20 | 0.21 |
| Threadgroup/Imageblock Store Utilization | 0.50 | 1.12 | 0.46 | 0.46 | 0.48 |
| F32 Utilization | 6.39 | 5.84 | 5.79 | 6.19 | 1.18 |
| F16 Utilization | 1.00 | 0.98 | 0.92 | 0.90 | 0.96 |
| **GPU ms / call** | 3.5905 | 4.0627 | 4.0183 | 3.7199 | 3.4181 |

## rowsize_4MiB

| counter | fused | fused_tiled | fused_rows4 | fused_rows8 | copy |
| --- | --- | --- | --- | --- | --- |
| GPU Read Bandwidth | 24.45 | 10.20 | 14.84 | 14.87 | 9.31 |
| GPU Write Bandwidth | 25.02 | 30.34 | 32.45 | 32.53 | 35.56 |
| ALU Utilization | 23.48 | 49.48 | 29.28 | 29.28 | 7.16 |
| ALU Limiter | 26.87 | 65.45 | 35.57 | 35.56 | 8.13 |
| Compute Occupancy | 77.70 | 79.79 | 77.49 | 77.29 | 69.24 |
| Buffer Read Limiter | 13.35 | 12.11 | 17.24 | 17.28 | 1.68 |
| Buffer Load Utilization | 13.00 | 11.83 | 17.00 | 17.02 | 1.34 |
| Buffer Write Limiter | 4.10 | 4.93 | 5.17 | 5.21 | 5.78 |
| Buffer Store Utilization | 3.74 | 4.56 | 4.88 | 4.89 | 5.35 |
| GPU Last Level Cache Utilization | 13.12 | 7.40 | 8.73 | 8.74 | 7.02 |
| GPU Last Level Cache Limiter | 19.10 | 8.77 | 20.30 | 20.60 | 32.92 |
| MMU Limiter | 22.45 | 8.79 | 32.31 | 32.87 | 39.59 |
| Threadgroup/Imageblock Load Limiter | 0.21 | 5.39 | 0.20 | 0.20 | 0.20 |
| Threadgroup/Imageblock Load Utilization | 0.20 | 4.69 | 0.19 | 0.19 | 0.19 |
| Threadgroup/Imageblock Store Limiter | 0.21 | 5.39 | 0.20 | 0.20 | 0.20 |
| Threadgroup/Imageblock Store Utilization | 0.48 | 1.06 | 0.46 | 0.45 | 0.46 |
| F32 Utilization | 4.89 | 5.71 | 5.97 | 5.97 | 1.13 |
| F16 Utilization | 0.96 | 1.03 | 0.91 | 0.90 | 0.91 |
| **GPU ms / call** | 5.1975 | 4.1896 | 3.9022 | 3.8843 | 3.4583 |

## rife_up4

| counter | fused | fused_tiled | fused_rows4 | fused_rows8 | copy |
| --- | --- | --- | --- | --- | --- |
| GPU Read Bandwidth | 1.28 | 1.14 | 1.25 | 1.25 | 1.33 |
| GPU Write Bandwidth | 22.47 | 17.86 | 22.12 | 22.01 | 24.90 |
| ALU Utilization | 34.85 | 45.01 | 34.03 | 33.77 | 11.78 |
| ALU Limiter | 51.44 | 62.04 | 50.31 | 50.04 | 13.57 |
| Compute Occupancy | 33.81 | 52.97 | 38.09 | 37.95 | 38.07 |
| Buffer Read Limiter | 16.23 | 9.38 | 15.89 | 15.75 | 1.66 |
| Buffer Load Utilization | 16.15 | 9.36 | 15.87 | 15.73 | 1.64 |
| Buffer Write Limiter | 5.92 | 3.67 | 5.79 | 5.74 | 6.54 |
| Buffer Store Utilization | 5.89 | 3.66 | 5.79 | 5.73 | 6.54 |
| GPU Last Level Cache Utilization | 19.65 | 6.34 | 11.14 | 9.31 | 7.53 |
| GPU Last Level Cache Limiter | 20.45 | 6.49 | 12.93 | 10.69 | 18.27 |
| MMU Limiter | 7.41 | 4.45 | 7.98 | 7.64 | 14.97 |
| Threadgroup/Imageblock Load Limiter | 0.36 | 3.98 | 0.35 | 0.35 | 0.36 |
| Threadgroup/Imageblock Load Utilization | 0.35 | 3.95 | 0.33 | 0.33 | 0.35 |
| Threadgroup/Imageblock Store Limiter | 0.36 | 3.98 | 0.35 | 0.35 | 0.36 |
| Threadgroup/Imageblock Store Utilization | 0.85 | 1.16 | 0.81 | 0.80 | 0.85 |
| F32 Utilization | 7.92 | 5.35 | 7.72 | 7.66 | 2.85 |
| F16 Utilization | 1.68 | 1.40 | 1.60 | 1.60 | 1.68 |
| **GPU ms / call** | 0.3851 | 0.5997 | 0.3873 | 0.3839 | 0.3292 |

## rife_up2

| counter | fused | fused_tiled | fused_rows4 | fused_rows8 | copy |
| --- | --- | --- | --- | --- | --- |
| GPU Read Bandwidth | 3.23 | 2.91 | 3.20 | 3.11 | 3.31 |
| GPU Write Bandwidth | 21.62 | 19.18 | 21.59 | 20.81 | 22.55 |
| ALU Utilization | 33.37 | 44.98 | 33.28 | 32.26 | 11.35 |
| ALU Limiter | 48.32 | 62.95 | 48.46 | 46.91 | 13.14 |
| Compute Occupancy | 32.90 | 54.97 | 37.72 | 37.10 | 40.85 |
| Buffer Read Limiter | 17.15 | 9.69 | 16.87 | 16.22 | 1.52 |
| Buffer Load Utilization | 16.72 | 9.66 | 16.82 | 16.17 | 1.48 |
| Buffer Write Limiter | 5.75 | 3.46 | 5.65 | 5.43 | 5.93 |
| Buffer Store Utilization | 5.59 | 3.44 | 5.62 | 5.41 | 5.90 |
| GPU Last Level Cache Utilization | 22.72 | 7.29 | 13.41 | 10.83 | 7.23 |
| GPU Last Level Cache Limiter | 24.08 | 7.52 | 16.17 | 13.21 | 20.41 |
| MMU Limiter | 9.25 | 4.72 | 11.53 | 10.88 | 19.45 |
| Threadgroup/Imageblock Load Limiter | 0.36 | 3.72 | 0.35 | 0.35 | 0.37 |
| Threadgroup/Imageblock Load Utilization | 0.35 | 3.69 | 0.34 | 0.34 | 0.35 |
| Threadgroup/Imageblock Store Limiter | 0.36 | 3.72 | 0.35 | 0.35 | 0.37 |
| Threadgroup/Imageblock Store Utilization | 0.84 | 1.26 | 0.81 | 0.81 | 0.85 |
| F32 Utilization | 7.60 | 4.93 | 7.56 | 7.34 | 2.78 |
| F16 Utilization | 1.67 | 1.23 | 1.62 | 1.61 | 1.68 |
| **GPU ms / call** | 0.3934 | 0.5716 | 0.3990 | 0.4013 | 0.3806 |

## GPU ms / call: v2 vs. v1

| config | variant | v2 | v1 |
| --- | --- | --- | --- |
| rowsize_1MiB | fused | 3.5905 | 2.9265 |
| rowsize_1MiB | fused_tiled | 4.0627 | 5.0848 |
| rowsize_1MiB | fused_rows4 | 4.0183 | - |
| rowsize_1MiB | fused_rows8 | 3.7199 | - |
| rowsize_1MiB | copy | 3.4181 | 3.6394 |
| rowsize_4MiB | fused | 5.1975 | 5.0906 |
| rowsize_4MiB | fused_tiled | 4.1896 | 4.8729 |
| rowsize_4MiB | fused_rows4 | 3.9022 | - |
| rowsize_4MiB | fused_rows8 | 3.8843 | - |
| rowsize_4MiB | copy | 3.4583 | 3.2042 |
| rife_up4 | fused | 0.3851 | 0.3450 |
| rife_up4 | fused_tiled | 0.5997 | 0.6298 |
| rife_up4 | fused_rows4 | 0.3873 | - |
| rife_up4 | fused_rows8 | 0.3839 | - |
| rife_up4 | copy | 0.3292 | 0.2271 |
| rife_up2 | fused | 0.3934 | - |
| rife_up2 | fused_tiled | 0.5716 | - |
| rife_up2 | fused_rows4 | 0.3990 | - |
| rife_up2 | fused_rows8 | 0.4013 | - |
| rife_up2 | copy | 0.3806 | - |
