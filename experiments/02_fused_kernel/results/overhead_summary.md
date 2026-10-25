# Fixed-cost summary

Source: `overhead.jsonl`

- Rows: 515 (0 invalid and dropped); workloads analyzed: 103
- `tables_only` is itself an `mx.eval` call, so it includes the per-call launch `floor`. `tables_net_ms` = `tables_only` - `floor` is the net table cost, and is what `saved_ms` (`fused` - `fused_cached`) should be compared against.
- Median saved time vs median net table time (all 103 workloads): 0.1051 ms vs 0.1263 ms (within 25%: True)
- Same comparison restricted to `fused` < 0.7 ms (51 workloads): 0.0902 ms vs 0.0959 ms (within 25%: True)
- Median launch `floor`: 0.2344 ms
- `fused` wins vs best other: 89 of 103; `fused_cached` wins: 103 of 103
- Of the 29 sweep (`speedup.csv`) ties/losses for uncached `fused`, 13 are still ties or losses for uncached `fused` in this rerun, and 16 flipped to wins with no code change, and 1 workload(s) that won in the sweep flipped the other way to a tie or loss here. Total non-wins in this rerun: 14. The noise at these sub-ms sizes goes both ways.
- Ties/losses for uncached `fused` in this rerun that become wins once tables are cached: 14 of 14
