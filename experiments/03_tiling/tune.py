"""Pick a tile size for `fused_tiled` by timing a small fixed workload set.

Upscaling is the only case with input reuse across nearby output pixels (see
README), so tiling can only help when scale > 1. Every workload here has
scale > 1.
"""

import argparse
import functools
import sys
from pathlib import Path

import mlx.core as mx
import pandas as pd

from mlx_upsample.bench import ResultStore, run_sweep, write_meta
from mlx_upsample.kernels.bilinear import fused, fused_tiled
from mlx_upsample.variants import gather
from mlx_upsample.workloads import Workload, grid, out_bytes, real

RESULTS = Path(__file__).parent / "results"
SEED = 0
MARGIN = 0.9  # same noise margin as 02_fused_kernel

TILE_SIZES = [(8, 2), (8, 4), (16, 4), (8, 8)]
VARIANTS = {
    "gather": gather,
    "fused": fused,
    **{f"tiled_{w}x{h}": functools.partial(fused_tiled, tile=(w, h)) for w, h in TILE_SIZES},
}

# Small/large H, each C, each scale, both dtypes: 2 * 3 * 2 * 2 = 24 is too
# many for a quick tuning run, so alternate dtype by index to halve it. Picked
# from grid()/real() by filtering, never by constructing Workload directly,
# so the 512 MB output cap (and --max-mb below) always applies.
_HSC = [(h, s, c) for h in (32, 128) for s in (2, 4) for c in (4, 32, 256)]
_DTYPE_BY_INDEX = {hsc: ("float32", "float16")[i % 2] for i, hsc in enumerate(_HSC)}


def _apply_cap(ws: list[Workload], max_mb: int) -> list[Workload]:
    capped = [w for w in ws if out_bytes(w) <= max_mb * 2**20]
    print(f"--max-mb={max_mb}: dropped {len(ws) - len(capped)} of {len(ws)} workloads")
    return capped


def workloads() -> list[Workload]:
    g = [w for w in grid() if (w.H, w.scale, w.C) in _DTYPE_BY_INDEX and w.dtype == _DTYPE_BY_INDEX[(w.H, w.scale, w.C)]]
    r = [w for w in real() if w.name.startswith("rife") and w.scale > 1 and w.B == 1]  # rife.up2, rife.up4: fp16, B=1
    return g + r


def analyze(path: Path) -> None:
    df = pd.read_json(path, lines=True, dtype=False)
    valid = df[df.valid == True]  # noqa: E712
    wide = valid.pivot_table(index=["name", "H", "C", "scale", "dtype"], columns="variant", values="ms_median").reset_index()
    tiles = [f"tiled_{w}x{h}" for w, h in TILE_SIZES]
    lines = ["# Tile tuning summary\n", f"Source: `{path.name}`\n", f"Workloads: {len(wide)}\n"]
    lines.append("\n| tile | median speedup vs fused | geomean speedup vs fused | wins | ties | losses |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for t in tiles:
        if t not in wide:
            continue
        d = wide.dropna(subset=["fused", t])
        if d.empty:
            lines.append(f"| {t} | (no data) | (no data) | 0 | 0 | 0 |")
            continue
        ratio = d.fused / d[t]  # >1 means tile faster than fused
        median = ratio.median()
        geomean = ratio.prod() ** (1 / len(ratio))
        wins = int((ratio > 1 / MARGIN).sum())
        losses = int((ratio < MARGIN).sum())
        ties = len(ratio) - wins - losses
        lines.append(f"| {t} | {median:.3f}x | {geomean:.3f}x | {wins} | {ties} | {losses} |")
    (path.parent / "tune_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {path.parent / 'tune_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-mb", type=int, default=256)
    args = ap.parse_args()

    # MLX keeps freed buffers for reuse; capping the cache returns memory to the OS while still
    # letting a 256 MB output reuse its buffer between timing iterations.
    mx.set_cache_limit(1 << 30)

    out = RESULTS / ("scratch/tune.jsonl" if args.smoke else "tune.jsonl")
    if not args.analyze:
        ws = _apply_cap(workloads(), args.max_mb)
        ws = ws[:2] if args.smoke else ws
        store = ResultStore(out)
        write_meta(out.parent / "tune_meta.json", vars(args), list(VARIANTS), SEED)
        run_sweep(store, ws, VARIANTS, ["fwd"])
    analyze(out)


if __name__ == "__main__":
    sys.exit(main())
