"""Time all variants on the grid and real workloads; append rows to results/sweep.jsonl."""

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

from mlx_upsample.bench import ResultStore, env_metadata, make_fwd, make_fwd_bwd, row_key, time_interleaved
from mlx_upsample.variants import VARIANTS
from mlx_upsample.workloads import grid, make_input, out_hw, real

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


def vjp_grad(f, x, sf, ac, cot):
    return mx.vjp(lambda a: f(a, sf, ac), [x], [cot.astype(f(x, sf, ac).dtype)])[1][0]


def maxdiff(a, b) -> float:
    return mx.abs(a.astype(mx.float32) - b).max().item()


def check(w, x, name, pas, ref, sf) -> dict:
    f = VARIANTS[name]
    y = f(x, sf, w.align_corners)
    diff = maxdiff(y, ref["y"])
    valid = diff <= (1e-5 if w.dtype == "float32" else 1e-2)
    row = {"out_shape": list(y.shape), "max_diff": diff}
    if pas == "fwd_bwd":
        gdiff = maxdiff(vjp_grad(f, x, sf, w.align_corners, ref["cot"]), ref["g"])
        row |= {"grad_max_diff": gdiff, "grad_rel_err": gdiff / ref["gmax"] if ref["gmax"] else 0.0}
        valid &= w.dtype != "float32" or row["grad_rel_err"] <= 1e-5  # float16 gradient error is recorded, not gated
    return row | {"valid": bool(valid)}


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
    (out.parent / "meta.json").write_text(json.dumps(
        {**env_metadata(), "args": {k: str(v) for k, v in vars(args).items()}, "variants": list(VARIANTS), "seed": SEED}, indent=2))

    total, n = len(ws) * len(passes) * len(VARIANTS), 0
    for i, w in enumerate(ws):
        sf = (w.scale, w.scale)
        x = make_input(w)
        for pas in passes:
            todo = [v for v in VARIANTS if not store.has(row_key({**asdict(w), "variant": v, "pass": pas}))]
            n += len(VARIANTS) - len(todo)
            if not todo:
                continue
            rows, ref = {}, {}
            try:
                y = VARIANTS["gather"](x, sf, w.align_corners)
                ref["y"] = y.astype(mx.float32)
                if pas == "fwd_bwd":
                    ref["cot"] = mx.random.normal(y.shape, key=mx.random.key(0)).astype(mx.float32)
                    ref["g"] = vjp_grad(VARIANTS["gather"], x.astype(mx.float32), sf, w.align_corners, ref["cot"])
                    ref["gmax"] = mx.abs(ref["g"]).max().item()
                mx.eval(ref)
            except Exception as e:  # reference failure: every variant gets an error row
                ref = {"error": f"reference: {e!r}"}
            fns = {}
            for v in todo:
                try:
                    if "error" in ref:
                        raise RuntimeError(ref["error"])
                    rows[v] = check(w, x, v, pas, ref, sf)
                    fns[v] = (make_fwd if pas == "fwd" else make_fwd_bwd)(VARIANTS[v], x, sf, w.align_corners)
                except Exception as e:
                    rows[v] = {"out_shape": None, "error": repr(e), "valid": False}
            if fns:
                try:
                    times = time_interleaved(fns, seed=i)
                except Exception as e:
                    times = {}
                    for v in fns:
                        rows[v] |= {"error": f"timing: {e!r}", "valid": False}
                for v, ms in times.items():
                    rows[v] |= {"ms_median": sorted(ms)[len(ms) // 2], "ms_repeats": ms}
            for v in todo:
                n += 1
                row = {**asdict(w), "variant": v, "pass": pas, **rows[v]}
                store.append(row)
                print(f"{n}/{total} {w.name} B={w.B} {w.dtype} {v} {pas} "
                      f"{row.get('ms_median', float('nan')):.3f} ms{'' if row['valid'] else ' INVALID'}", flush=True)
        x = ref = y = fns = rows = None
        mx.clear_cache()


if __name__ == "__main__":
    sys.exit(main())
