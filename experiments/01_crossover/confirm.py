"""Re-time key configs in fresh processes (one timing per process) to check the sweep."""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from mlx_upsample.bench import KEY_FIELDS, row_key
from mlx_upsample.workloads import Workload, grid, real

RESULTS = Path(__file__).parent / "results"
VARIANTS = ("gather", "matmul", "pr4418")
PASSES = ("fwd", "fwd_bwd")


def one(spec: dict) -> None:
    from mlx_upsample.bench import make_fwd, make_fwd_bwd, time_interleaved
    from mlx_upsample.variants import VARIANTS as V
    from mlx_upsample.workloads import make_input

    w = Workload(**spec["w"])
    x = make_input(w)
    make = make_fwd if spec["pass"] == "fwd" else make_fwd_bwd
    fns = {v: make(V[v], x, (w.scale, w.scale), w.align_corners) for v in VARIANTS}
    ms = time_interleaved(fns, seed=spec["run"])
    print(json.dumps({v: sorted(t)[len(t) // 2] for v, t in ms.items()}))


def failed_in_sweep() -> set[tuple]:
    """(name, dtype, B, pass) that errored or were invalid in the sweep, for example an out-of-memory run."""
    path = RESULTS / "sweep.jsonl"
    bad = set()
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("error") or not r["valid"]:
            bad.add((r["name"], r["dtype"], r["B"], r["pass"]))
    return bad


def targets(args) -> list[tuple[Workload, str]]:
    out = _targets(args)
    bad = failed_in_sweep()
    skipped = [(w, p) for w, p in out if (w.name, w.dtype, w.B, p) in bad]
    for w, p in skipped:
        print(f"skip {w.name} B={w.B} {w.dtype} {p}: error or invalid in the sweep")
    return [t for t in out if t not in skipped]


def _targets(args) -> list[tuple[Workload, str]]:
    ws = {(w.name, w.dtype, w.B): w for w in grid() + real()}.values()
    if args.smoke:
        return [(w, "fwd") for w in ws if w.name == "grid.h64.s2.c4" and w.dtype == "float32"]
    if args.names:
        return [(w, p) for w in ws if w.name in args.names for p in PASSES]
    out = [(w, p) for w in ws if w.name.startswith("rife.") for p in PASSES]
    csv = RESULTS / "crossover.csv"
    if csv.exists():
        sizes = sorted({w.H for w in ws if w.set == "grid"})
        for r in pd.read_csv(csv).dropna(subset=["last_win_H"]).to_dict("records"):
            lo, hi = int(r["first_win_H"]), int(r["last_win_H"])
            edges = [hi, *sizes[sizes.index(hi) + 1 :][:1]]  # last matmul win and the next size above
            if lo > sizes[0]:
                edges += [lo, sizes[sizes.index(lo) - 1]]  # first matmul win and the size below
            for n in edges:
                out += [(w, r["pass"]) for w in ws if (w.set, w.H, w.scale, w.C, w.dtype) == ("grid", n, r["scale"], r["C"], r["dtype"])]
    return list(dict.fromkeys(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--names", nargs="*")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--one", help="internal: JSON spec of one config to time")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.one:
        return one(json.loads(args.one))

    path = args.out or RESULTS / ("scratch/confirm.jsonl" if args.smoke else "confirm.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    done, tail_ok = set(), True
    if path.exists():
        text = path.read_text()
        for line in text.splitlines():
            try:
                r = json.loads(line)
                done.add((*row_key(r), r["run"]))
            except (json.JSONDecodeError, KeyError):
                pass  # truncated last line from a crash
        tail_ok = not text or text.endswith("\n")
    runs = 2 if args.smoke else args.runs
    todo = targets(args)
    print(f"{len(todo)} workload-pass targets x {runs} runs (all variants in one process)")
    for w, p in todo:
        for run in range(runs):
            keys = {v: (*(({**asdict(w), "variant": v, "pass": p})[k] for k in KEY_FIELDS), run) for v in VARIANTS}
            if all(k in done for k in keys.values()):
                continue
            spec = json.dumps({"w": asdict(w), "pass": p, "run": run})
            res = subprocess.run([sys.executable, __file__, "--one", spec], capture_output=True, text=True)
            if res.returncode:
                print(f"FAILED {w.name} {p}: {res.stderr[-200:]}")
                continue
            ms = json.loads(res.stdout.splitlines()[-1])
            with path.open("a") as f:
                if not tail_ok:
                    f.write("\n")
                    tail_ok = True
                for v in VARIANTS:
                    f.write(json.dumps({**asdict(w), "variant": v, "pass": p, "run": run, "ms_median": ms[v]}) + "\n")
            done.update(keys.values())
            print(f"{w.name} B={w.B} {w.dtype} {p} run {run}: " + " ".join(f"{v}={ms[v]:.3f}" for v in VARIANTS), flush=True)


if __name__ == "__main__":
    main()
