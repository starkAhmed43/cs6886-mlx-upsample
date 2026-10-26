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

| counter | fused | fused_tiled | copy |
| --- | --- | --- | --- |
| GPU Read Bandwidth | 42.42 | 9.45 | 8.64 |
| GPU Write Bandwidth | 41.21 | 25.24 | 34.19 |
| ALU Utilization | 33.22 | 59.96 | 3.36 |
| ALU Limiter | 40.02 | 89.95 | 3.38 |
| Compute Occupancy | 80.81 | 64.24 | 75.96 |
| Buffer Read Limiter | 22.40 | 10.84 | 1.60 |
| Buffer Load Utilization | 22.10 | 10.61 | 1.32 |
| Buffer Write Limiter | 6.65 | 4.14 | 5.61 |
| Buffer Store Utilization | 6.34 | 3.86 | 5.26 |
| GPU Last Level Cache Utilization | 17.20 | 6.23 | 6.62 |
| GPU Last Level Cache Limiter | 35.11 | 6.45 | 35.98 |
| MMU Limiter | 47.20 | 5.65 | 43.20 |
| Threadgroup/Imageblock Load Limiter | 0.01 | 3.99 | 0.00 |
| Threadgroup/Imageblock Load Utilization | 0.00 | 3.84 | 0.00 |
| Threadgroup/Imageblock Store Limiter | 0.01 | 3.99 | 0.00 |
| Threadgroup/Imageblock Store Utilization | 0.01 | 0.55 | 0.01 |
| F32 Utilization | 6.33 | 7.48 | 0.02 |
| F16 Utilization | 0.01 | 0.06 | 0.01 |
| **GPU ms / call** | 2.9265 | 5.0848 | 3.6394 |

## rowsize_4MiB

| counter | fused | fused_tiled | copy |
| --- | --- | --- | --- |
| GPU Read Bandwidth | 25.24 | 9.30 | 10.55 |
| GPU Write Bandwidth | 24.70 | 26.77 | 38.57 |
| ALU Utilization | 20.88 | 59.43 | 4.22 |
| ALU Limiter | 23.53 | 87.93 | 4.26 |
| Compute Occupancy | 76.59 | 63.36 | 71.86 |
| Buffer Read Limiter | 14.30 | 11.17 | 1.99 |
| Buffer Load Utilization | 13.14 | 10.95 | 1.49 |
| Buffer Write Limiter | 4.02 | 4.29 | 6.54 |
| Buffer Store Utilization | 3.67 | 4.03 | 5.90 |
| GPU Last Level Cache Utilization | 13.02 | 6.38 | 7.67 |
| GPU Last Level Cache Limiter | 19.83 | 7.07 | 34.35 |
| MMU Limiter | 23.18 | 6.10 | 41.90 |
| Threadgroup/Imageblock Load Limiter | 0.10 | 4.21 | 0.06 |
| Threadgroup/Imageblock Load Utilization | 0.09 | 4.05 | 0.05 |
| Threadgroup/Imageblock Store Limiter | 0.10 | 4.21 | 0.06 |
| Threadgroup/Imageblock Store Utilization | 0.21 | 0.57 | 0.11 |
| F32 Utilization | 3.92 | 7.35 | 0.09 |
| F16 Utilization | 0.37 | 0.13 | 0.13 |
| **GPU ms / call** | 5.0906 | 4.8729 | 3.2042 |

## rife_up4

| counter | fused | fused_tiled | copy |
| --- | --- | --- | --- |
| GPU Read Bandwidth | 1.76 | 0.78 | 1.16 |
| GPU Write Bandwidth | 23.40 | 17.09 | 31.30 |
| ALU Utilization | 32.30 | 42.23 | 6.45 |
| ALU Limiter | 47.65 | 64.43 | 6.47 |
| Compute Occupancy | 34.98 | 54.45 | 39.25 |
| Buffer Read Limiter | 16.51 | 8.74 | 2.13 |
| Buffer Load Utilization | 16.39 | 8.73 | 2.11 |
| Buffer Write Limiter | 6.02 | 3.25 | 8.45 |
| Buffer Store Utilization | 5.98 | 3.25 | 8.45 |
| GPU Last Level Cache Utilization | 20.06 | 5.26 | 9.07 |
| GPU Last Level Cache Limiter | 21.44 | 5.44 | 18.88 |
| MMU Limiter | 8.01 | 3.73 | 18.73 |
| Threadgroup/Imageblock Load Limiter | 0.32 | 3.29 | 0.02 |
| Threadgroup/Imageblock Load Utilization | 0.27 | 3.28 | 0.02 |
| Threadgroup/Imageblock Store Limiter | 0.32 | 3.29 | 0.02 |
| Threadgroup/Imageblock Store Utilization | 0.56 | 0.26 | 0.03 |
| F32 Utilization | 7.06 | 4.70 | 1.07 |
| F16 Utilization | 1.03 | 0.05 | 0.04 |
| **GPU ms / call** | 0.3450 | 0.6298 | 0.2271 |

## rife_up2

| counter |  |
| --- |  |
| GPU Read Bandwidth |  |
| GPU Write Bandwidth |  |
| ALU Utilization |  |
| ALU Limiter |  |
| Compute Occupancy |  |
| Buffer Read Limiter |  |
| Buffer Load Utilization |  |
| Buffer Write Limiter |  |
| Buffer Store Utilization |  |
| GPU Last Level Cache Utilization |  |
| GPU Last Level Cache Limiter |  |
| MMU Limiter |  |
| Threadgroup/Imageblock Load Limiter |  |
| Threadgroup/Imageblock Load Utilization |  |
| Threadgroup/Imageblock Store Limiter |  |
| Threadgroup/Imageblock Store Utilization |  |
| F32 Utilization |  |
| F16 Utilization |  |
| **GPU ms / call** |  |

## GPU ms / call: v1 vs. v1

| config | variant | v1 | v1 |
| --- | --- | --- | --- |
| rowsize_1MiB | fused | 2.9265 | 2.9265 |
| rowsize_1MiB | fused_tiled | 5.0848 | 5.0848 |
| rowsize_1MiB | copy | 3.6394 | 3.6394 |
| rowsize_4MiB | fused | 5.0906 | 5.0906 |
| rowsize_4MiB | fused_tiled | 4.8729 | 4.8729 |
| rowsize_4MiB | copy | 3.2042 | 3.2042 |
| rife_up4 | fused | 0.3450 | 0.3450 |
| rife_up4 | fused_tiled | 0.6298 | 0.6298 |
| rife_up4 | copy | 0.2271 | 0.2271 |
