"""Time all variants on the grid and real workloads; append rows to results/sweep.jsonl."""

import argparse
import random
import sys
from pathlib import Path

from mlx_upsample.bench import ResultStore, run_sweep, write_meta
from mlx_upsample.variants import VARIANTS
from mlx_upsample.workloads import grid, real

RESULTS = Path(__file__).parent / "results"
SEED = 0
SMOKE_GRID = [(64, 2, 4, "float32"), (128, 2, 32, "float16"), (192, 4, 4, "float32"), (64, 0.5, 32, "float32"), (256, 0.25, 4, "float16")]
SMOKE_REAL = {"pspnet.bin6", "pspnet.bin2", "upernet.fpn16"}


def pick_workloads(args):
    ws = (grid() if args.set in ("grid", "all") else []) + (real() if args.set in ("real", "all") else [])
    if args.smoke:
        ws = [w for w in grid() if (w.H, w.scale, w.C, w.dtype) in SMOKE_GRID] + [
            w for w in real() if w.name in SMOKE_REAL and w.B == 1
        ]
    random.Random(SEED).shuffle(ws)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["grid", "real", "all"], default="all")
    ap.add_argument("--pass", dest="passes", choices=["fwd", "fwd_bwd", "both"], default="both")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out = args.out or RESULTS / ("scratch/sweep.jsonl" if args.smoke else "sweep.jsonl")
    store = ResultStore(out)
    passes = ["fwd", "fwd_bwd"] if args.passes == "both" else [args.passes]
    ws = pick_workloads(args)
    write_meta(out.parent / "meta.json", vars(args), list(VARIANTS), SEED)

    run_sweep(store, ws, VARIANTS, passes)


if __name__ == "__main__":
    sys.exit(main())
