"""Forward parity of the fused Metal kernel against stock mlx.nn.Upsample.

Runs over every workload in mlx_upsample.workloads (grid and real), writes
results/parity.csv and results/parity_summary.md. No timing here; see 01_crossover
for speed.
"""

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from mlx_upsample.kernels.bilinear import fused_uncached
from mlx_upsample.workloads import grid, make_input, out_hw, real

RESULTS = Path(__file__).parent / "results"
REL_ERR_LIMIT = 1e-5
SMOKE_GRID = [(64, 2, 4, "float32"), (128, 2, 32, "float16"), (192, 4, 4, "float32"),
              (64, 0.5, 32, "float32"), (256, 0.25, 4, "float16")]
SMOKE_REAL = {"pspnet.bin6", "pspnet.bin2", "upernet.fpn16"}

FIELDS = ["name", "set", "B", "H", "W", "C", "scale", "align_corners", "dtype",
          "out_shape", "dtype_match", "shape_match", "max_abs_diff", "rel_err", "pass"]


def pick_workloads(smoke: bool) -> list:
    ws = grid() + real()
    if smoke:
        ws = [w for w in grid() if (w.H, w.scale, w.C, w.dtype) in SMOKE_GRID] + [
            w for w in real() if w.name in SMOKE_REAL and w.B == 1
        ]
    return ws


def check_one(w) -> dict:
    x = make_input(w)
    sf = (w.scale, w.scale)
    ref = nn.Upsample(scale_factor=sf, mode="linear", align_corners=w.align_corners)(x)
    out = fused_uncached(x, sf, w.align_corners)
    mx.eval(ref, out)

    shape_match = out.shape == ref.shape
    dtype_match = out.dtype == ref.dtype
    max_abs_diff = mx.abs(out.astype(mx.float32) - ref.astype(mx.float32)).max().item() if shape_match else float("nan")
    ref_max = mx.abs(ref.astype(mx.float32)).max().item() if shape_match else 0.0
    rel_err = max_abs_diff / ref_max if shape_match and ref_max > 0 else max_abs_diff
    passed = bool(shape_match and dtype_match and rel_err <= REL_ERR_LIMIT)

    row = {
        "name": w.name, "set": w.set, "B": w.B, "H": w.H, "W": w.W, "C": w.C,
        "scale": w.scale, "align_corners": w.align_corners, "dtype": w.dtype,
        "out_shape": "x".join(map(str, out.shape)), "dtype_match": dtype_match,
        "shape_match": shape_match, "max_abs_diff": max_abs_diff, "rel_err": rel_err,
        "pass": passed,
    }
    del x, ref, out
    return row


def write_summary(rows: list[dict], path: Path):
    lines = [f"# Parity summary ({len(rows)} workloads)", ""]
    for dtype in ("float32", "float16"):
        sub = [r for r in rows if r["dtype"] == dtype]
        if not sub:
            continue
        n_pass = sum(r["pass"] for r in sub)
        worst = max(sub, key=lambda r: r["rel_err"])
        lines.append(f"- {dtype}: {n_pass}/{len(sub)} pass, worst rel_err {worst['rel_err']:.3e} ({worst['name']})")
    failures = [r for r in rows if not r["pass"]]
    lines.append("")
    if failures:
        lines.append(f"## Failures ({len(failures)})")
        for r in failures:
            lines.append(f"- {r['name']} {r['dtype']} scale={r['scale']} align_corners={r['align_corners']}: "
                          f"shape_match={r['shape_match']} dtype_match={r['dtype_match']} rel_err={r['rel_err']:.3e}")
    else:
        lines.append("No failures.")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS / "scratch" if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)

    ws = pick_workloads(args.smoke)
    rows = []
    for i, w in enumerate(ws, 1):
        row = check_one(w)
        rows.append(row)
        print(f"{i}/{len(ws)} {w.name} B={w.B} {w.dtype} scale={w.scale} "
              f"rel_err={row['rel_err']:.2e}{'' if row['pass'] else ' FAIL'}", flush=True)
        mx.clear_cache()

    with open(out_dir / "parity.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    write_summary(rows, out_dir / "parity_summary.md")


if __name__ == "__main__":
    sys.exit(main())
