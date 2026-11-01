"""Roofline analysis built only from `results/sweep.jsonl` -- no new timing.

Reuses the shared model in `mlx_upsample.roofline` (see its docstring for the hardware,
traffic and FLOPs assumptions). All four variants (`gather`, `gather_compiled`, `matmul`,
`pr4418`) upcast their output to float32 regardless of input dtype (verified against
`mlx_upsample.variants`), so `out_elem` is always 4 bytes here.

FLOPs differ by method, not by hardware: `gather`, `gather_compiled`, and `pr4418` when it
takes its gather path (`uses_matmul` is false) use the elementwise bilinear model (9
flops/output-element/pass). `matmul` always runs the dense per-axis matmul (it is called
directly, bypassing PR #4418's upscale-only gate), and `pr4418` uses it too whenever the gate
(`uses_matmul`) picks the matmul path. This is the point of this experiment's roofline: matmul
does far more arithmetic for the same output, so it shows up at much higher achieved GFLOP/s but
not higher achieved GB/s.
"""

import argparse
from pathlib import Path

import pandas as pd

from mlx_upsample.roofline import PEAK_BW_GBPS, PEAK_COMPUTE_GFLOPS, build_table, elementwise_flops, figure, matmul_flops, min_bytes, uses_matmul

RESULTS = Path(__file__).parent / "results"
VARIANT_COLORS = {"gather": "tab:orange", "gather_compiled": "tab:red", "matmul": "tab:blue", "pr4418": "tab:green"}
PASS_MARKERS = {"fwd": "o", "fwd_bwd": "^"}
VARIANT_ORDER = ["gather", "gather_compiled", "matmul", "pr4418"]


def bytes_flops(row: pd.Series) -> tuple[int, int]:
    is_matmul = row["variant"] == "matmul" or (row["variant"] == "pr4418" and uses_matmul(row))
    flops = matmul_flops(row) if is_matmul else elementwise_flops(row)
    return min_bytes(row, 4), flops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=RESULTS / "sweep.jsonl")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    args = ap.parse_args()

    df = pd.read_json(args.inp, lines=True, dtype=False)
    valid = df[df.valid == True]  # noqa: E712
    print(f"{len(df)} rows, dropped {len(df) - len(valid)} invalid")
    rf = build_table(valid, bytes_flops)

    fig_dir = args.out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rf.to_csv(args.out_dir / "roofline.csv", index=False)
    figure(rf, fig_dir / "roofline.png", VARIANT_COLORS, VARIANT_ORDER, PASS_MARKERS)

    print(f"AI range: {rf.ai.min():.3g} - {rf.ai.max():.3g} FLOP/byte (ridge point {PEAK_COMPUTE_GFLOPS/PEAK_BW_GBPS:.3g})")
    print(rf.groupby(["variant", "pass"]).frac_of_peak_bw.median())
    print(f"rows above 1.0 frac_of_peak_bw: {int((rf.frac_of_peak_bw > 1.0).sum())}")
    print(f"wrote {args.out_dir / 'roofline.csv'} and {fig_dir / 'roofline.png'}")


if __name__ == "__main__":
    main()
