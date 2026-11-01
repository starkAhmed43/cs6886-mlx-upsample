"""Roofline analysis built only from `results/sweep.jsonl` -- no new timing, forward+backward.

Reuses the shared model in `mlx_upsample.roofline` (see its docstring). All four variants
(`gather`, `matmul`, `pr4418`, `fused`) upcast their forward output to float32 regardless of
input dtype. `fused` is `mlx_upsample.kernels.bilinear.upsample`: its forward kernel is the
same `_launch` as `02_fused_kernel`'s `fused` (float32 output), and only its backward kernel
casts the grad-input back to the input dtype -- which the shared model already does uniformly
for every variant (the grad-input write always uses the input dtype's size). So `out_elem` is
always 4 bytes for the forward/grad-output bytes here, same as `01_crossover`/`02_fused_kernel`.

FLOPs: `fused` is elementwise (9 flops/output-element/pass; its backward is a hand-written
gather-style accumulation, not a matmul). `matmul` is called directly, bypassing PR #4418's
upscale-only gate, so it always gets the dense matmul model, including for downscale rows.
`pr4418` gets the matmul model only where its own gate (`uses_matmul`) picks that path (upscale,
H and W <= 128); every downscale row therefore uses the elementwise model for `pr4418`.
"""

import argparse
from pathlib import Path

import pandas as pd

from mlx_upsample.roofline import PEAK_BW_GBPS, PEAK_COMPUTE_GFLOPS, build_table, elementwise_flops, figure, matmul_flops, min_bytes, uses_matmul

RESULTS = Path(__file__).parent / "results"
VARIANT_COLORS = {"gather": "tab:orange", "matmul": "tab:blue", "pr4418": "tab:green", "fused": "tab:purple"}
VARIANT_ORDER = ["gather", "matmul", "pr4418", "fused"]


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
    figure(rf, fig_dir / "roofline.png", VARIANT_COLORS, VARIANT_ORDER)

    print(f"AI range: {rf.ai.min():.3g} - {rf.ai.max():.3g} FLOP/byte (ridge point {PEAK_COMPUTE_GFLOPS/PEAK_BW_GBPS:.3g})")
    print(rf.groupby(["variant", "pass"]).frac_of_peak_bw.median())
    print(f"fused max frac_of_peak_bw: {rf[rf.variant == 'fused'].frac_of_peak_bw.max():.3f}")
    print(f"rows above 1.0 frac_of_peak_bw: {int((rf.frac_of_peak_bw > 1.0).sum())}")
    print(f"wrote {args.out_dir / 'roofline.csv'} and {fig_dir / 'roofline.png'}")


if __name__ == "__main__":
    main()
