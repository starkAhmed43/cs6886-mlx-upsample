# Post-profiling kernel-change summary

Source: `improve.jsonl`

Workloads: 64


| variant | wins | ties | losses | median ratio | min ratio | max ratio |
| --- | --- | --- | --- | --- | --- | --- |
| fused_rows4 | 6 | 56 | 2 | 0.994x | 0.857x | 1.375x |
| fused_rows8 | 3 | 56 | 5 | 0.993x | 0.833x | 1.400x |
| fused_tiled | 2 | 18 | 44 | 0.779x | 0.606x | 1.376x |

## By workload set


| variant | set | n | wins | ties | losses | median ratio |
| --- | --- | --- | --- | --- | --- | --- |
| fused_rows4 | grid | 12 | 1 | 10 | 1 | 1.000x |
| fused_rows4 | real | 49 | 3 | 45 | 1 | 0.992x |
| fused_rows4 | rowsize | 3 | 2 | 1 | 0 | 1.165x |
| fused_rows8 | grid | 12 | 0 | 11 | 1 | 1.013x |
| fused_rows8 | real | 49 | 1 | 44 | 4 | 0.987x |
| fused_rows8 | rowsize | 3 | 2 | 1 | 0 | 1.186x |
| fused_tiled | grid | 12 | 0 | 7 | 5 | 0.908x |
| fused_tiled | real | 49 | 0 | 11 | 38 | 0.756x |
| fused_tiled | rowsize | 3 | 2 | 0 | 1 | 1.123x |

## Every rowsize and rife workload


| name | set | B | H | W | C | scale | dtype | fused | fused_rows4 | fused_rows8 | fused_tiled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rife.up2 | real | 1 | 384 | 640 | 4 | 2.000000001 | float16 | 0.7956 | 0.7571 | 0.7916 | 1.3016 |
| rife.up2 | real | 8 | 384 | 640 | 4 | 2.000000001 | float32 | 3.8055 | 3.9453 | 3.9969 | 4.1755 |
| rife.up4 | real | 1 | 192 | 320 | 4 | 4.000000001 | float16 | 0.7103 | 0.7567 | 0.7168 | 1.1730 |
| rife.up4 | real | 8 | 192 | 320 | 4 | 4.000000001 | float32 | 3.3893 | 3.6126 | 3.4018 | 3.6769 |
| rowsize.s2.1MiB | rowsize | 1 | 32 | 4096 | 64 | 2.0 | float32 | 4.2471 | 4.6141 | 4.5410 | 4.8339 |
| rowsize.s2.4MiB | rowsize | 1 | 8 | 16384 | 64 | 2.0 | float32 | 6.2028 | 4.5096 | 4.4316 | 4.5086 |
| rowsize.s4.4MiB | rowsize | 1 | 2 | 16384 | 64 | 4.0 | float32 | 4.4368 | 3.8081 | 3.7417 | 3.9513 |
