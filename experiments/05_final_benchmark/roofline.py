"""Roofline analysis built only from `results/table.csv` -- no new timing.

Hardware model (Apple M1, 8-core GPU; constants below are the only inputs that
are not derived from the existing results):

- DRAM bandwidth peak: 68.25 GB/s.
- fp32 compute peak: 2.6 TFLOP/s.
- Ridge point (compute/bandwidth): 2.6e12 / 68.25e9 ~= 38.1 FLOP/byte.

Traffic model (minimum bytes a correct implementation must move, used for all
three variants so the plot shows wasted traffic as lower achieved GB/s, not as
a different roofline):

- forward: read input once, write output once. For the input read, a correct
  bilinear kernel only touches at most 2 input indices per output index on
  each axis, so for downscaling (oh < H or ow < W) the forward input read is
  capped at `min(H, 2*oh) * min(W, 2*ow)` pixels per (batch, channel), not the
  full `H*W`. For upscaling this cap is not binding (every input pixel is
  read at least once), so it has no effect.
- forward+backward: forward bytes, plus read grad-output once (same size as
  the output) and write grad-input once. The grad-input write always covers
  the full input shape (every input position gets a gradient, even where it
  is zero), so it is not subject to the downscaling read cap above.
- Element size is 2 bytes for float16, 4 for float32. `fused_cpp` keeps the
  input dtype on output (so its output/grad-output bytes use the input
  dtype's size); `stock` and `pr4418` upcast output to float32 regardless of
  input dtype (see the out_dtype table in `summary.md`), so their
  output/grad-output bytes always use 4-byte elements. Input/grad-input bytes
  always use the input dtype's size for all three variants.
- Cache-line granularity (e.g. a kernel reading a full 32/64/128-byte line to
  fetch one small element) is ignored; bytes are counted at the element
  level only.

FLOPs model (bilinear interpolation): each output element is 3 lerps, 3 flops
each (sub, mul, add) = 9 flops/output-element for the forward pass. The
backward pass is treated as the same cost per output element (4 weighted
accumulations into the input gradient, ~8-9 flops), so `fwd_bwd` rows use
9 flops/output-element for each of the two passes (18 total). The algorithm
is the same for all three variants, so FLOPs only depend on workload shape
and pass, not on variant; only the achieved GFLOP/s and GB/s differ.
"""

import argparse
from pathlib import Path

import pandas as pd

from mlx_upsample.roofline import (
    PEAK_BW_GBPS,
    PEAK_COMPUTE_GFLOPS,
    build_table as _build_table,
    dtype_bytes,
    elementwise_flops,
    figure,
    min_bytes,
    single_panel_figure,
)

RESULTS = Path(__file__).parent / "results"
VARIANT_COLORS = {"stock": "tab:orange", "pr4418": "tab:green", "fused_cpp": "tab:blue"}
PASS_MARKERS = {"fwd": "o", "fwd_bwd": "^"}
VARIANT_ORDER = ["stock", "pr4418", "fused_cpp"]


def _out_elem(row: pd.Series) -> int:
    # fused_cpp (mx.fast.upsample_bilinear) preserves the input dtype; stock/pr4418 upcast to float32.
    return dtype_bytes(row["dtype"]) if row["variant"] == "fused_cpp" else 4


def min_bytes_and_flops(row: pd.Series) -> tuple[int, int]:
    """Minimum bytes moved and FLOPs for one (workload, pass, variant) row."""
    return min_bytes(row, _out_elem(row)), elementwise_flops(row)


def build_table(table: pd.DataFrame) -> pd.DataFrame:
    return _build_table(table, min_bytes_and_flops)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=RESULTS / "table.csv")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    table = pd.read_csv(args.inp)
    rf = build_table(table)

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rf.to_csv(args.out_dir / "roofline.csv", index=False)
    figure(rf, fig_dir / "roofline.png", VARIANT_COLORS, VARIANT_ORDER, PASS_MARKERS)
    single_panel_figure(rf, fig_dir / "roofline_single.png", VARIANT_COLORS, PASS_MARKERS)

    print(f"AI range: {rf.ai.min():.3g} - {rf.ai.max():.3g} FLOP/byte (ridge point {PEAK_COMPUTE_GFLOPS/PEAK_BW_GBPS:.3g})")
    print(rf.groupby(["variant", "pass"]).frac_of_peak_bw.median())
    print(f"fused_cpp max frac_of_peak_bw: {rf[rf.variant == 'fused_cpp'].frac_of_peak_bw.max():.3f}")
    print(f"wrote {args.out_dir / 'roofline.csv'}, {fig_dir / 'roofline.png'} and {fig_dir / 'roofline_single.png'}")


if __name__ == "__main__":
    main()
