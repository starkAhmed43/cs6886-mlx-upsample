"""Do the post-profiling kernel changes help, under the project's normal interleaved timing?

GPU profiling (`profile.py`) suggested two changes to `fused`/`fused_tiled`:
taller threadgroups (`tg_rows=4|8`, default stays 1) and a division-free
cooperative load in `fused_tiled`. This script re-times all four variants
together (`fused`, `fused_rows4`, `fused_rows8`, `fused_tiled`) with
`time_interleaved`, the same timing method `tune.py`/`sweep.py` use, to
confirm the profiler's read on real, scheduled timings rather than trace
counters.

No `gather` baseline here (memory, not speed, is `gather`'s problem -- see
`02_fused_kernel`): every variant is checked against `fused` itself instead.
"""

import argparse
import functools
import sys
from dataclasses import asdict
from pathlib import Path

import mlx.core as mx

sys.path.insert(0, str(Path(__file__).parent))
from tune import workloads as tune_workloads  # noqa: E402

from mlx_upsample.bench import ResultStore, make_fwd, maxdiff, row_key, time_interleaved, write_meta  # noqa: E402
from mlx_upsample.kernels.bilinear import fused, fused_tiled  # noqa: E402
from mlx_upsample.workloads import Workload, make_input, out_bytes, real  # noqa: E402

RESULTS = Path(__file__).parent / "results"
SEED = 0
MARGIN = 0.9  # same noise margin as 02_fused_kernel/03_tiling

VARIANTS = {
    "fused": fused,
    "fused_rows4": functools.partial(fused, tg_rows=4),
    "fused_rows8": functools.partial(fused, tg_rows=8),
    "fused_tiled": fused_tiled,
}

# Three fixed row-size configs (see rowsize.py): scale 2, 1 MiB row; scale 2,
# 4 MiB row; scale 4, 4 MiB row. Each holds C=64, B=1, align_corners=False.
ROWSIZE_CONFIGS = [
    ("rowsize.s2.1MiB", 2, 32, 4096),
    ("rowsize.s2.4MiB", 2, 8, 16384),
    ("rowsize.s4.4MiB", 4, 2, 16384),
]


def rowsize_workloads() -> list[Workload]:
    return [
        Workload(name, "rowsize.py fixed config", "rowsize", 1, h, w, 64, scale, False, "float32")
        for name, scale, h, w in ROWSIZE_CONFIGS
    ]


def _apply_cap(ws: list[Workload], max_mb: int) -> list[Workload]:
    capped = [w for w in ws if out_bytes(w) <= max_mb * 2**20]
    print(f"--max-mb={max_mb}: dropped {len(ws) - len(capped)} of {len(ws)} real workloads")
    return capped


def pick_workloads(max_mb: int) -> list[Workload]:
    r = _apply_cap([w for w in real() if w.scale > 1], max_mb)
    t = tune_workloads()
    rs = rowsize_workloads()
    seen, out = set(), []
    for w in [*r, *t, *rs]:
        if w not in seen:
            seen.add(w)
            out.append(w)
    return out


def run(store: ResultStore, workloads: list[Workload], print_progress: bool = True) -> None:
    total, n = len(workloads) * len(VARIANTS), 0
    for i, w in enumerate(workloads):
        mx.set_cache_limit(1 << 30)
        sf = (w.scale, w.scale)
        todo = [v for v in VARIANTS if not store.has(row_key({**asdict(w), "variant": v, "pass": "fwd"}))]
        n += len(VARIANTS) - len(todo)
        if todo:
            x = make_input(w)
            try:
                ref = fused(x, sf, w.align_corners).astype(mx.float32)
                mx.eval(ref)
            except Exception as e:
                ref = None
                ref_err = f"reference: {e!r}"
            tol = 1e-5 if w.dtype == "float32" else 1e-2
            rows, fns = {}, {}
            for v in todo:
                try:
                    if ref is None:
                        raise RuntimeError(ref_err)
                    y = VARIANTS[v](x, sf, w.align_corners)
                    diff = maxdiff(y, ref)
                    rows[v] = {"max_diff": diff, "valid": bool(diff <= tol)}
                    fns[v] = make_fwd(VARIANTS[v], x, sf, w.align_corners)
                except Exception as e:
                    rows[v] = {"max_diff": None, "valid": False, "error": repr(e)}
            if fns:
                try:
                    times = time_interleaved(fns, seed=i)
                    for v, ms in times.items():
                        rows[v] |= {"ms_median": sorted(ms)[len(ms) // 2], "ms_repeats": ms}
                except Exception as e:
                    for v in fns:
                        rows[v] |= {"error": f"timing: {e!r}", "valid": False}
            for v in todo:
                n += 1
                row = {**asdict(w), "variant": v, "pass": "fwd", **rows[v]}
                store.append(row)
                if print_progress:
                    print(f"{n}/{total} {w.name} B={w.B} {w.dtype} {v} "
                          f"{row.get('ms_median', float('nan')):.3f} ms{'' if row['valid'] else ' INVALID'}", flush=True)
            x = ref = fns = None
        mx.clear_cache()


def analyze(path: Path) -> None:
    import pandas as pd

    df = pd.read_json(path, lines=True, dtype=False)
    valid = df[df.valid == True]  # noqa: E712
    wide = valid.pivot_table(
        index=["name", "set", "B", "H", "W", "C", "scale", "dtype"], columns="variant", values="ms_median"
    ).reset_index()
    tested = [v for v in VARIANTS if v != "fused"]
    lines = ["# Post-profiling kernel-change summary\n", f"Source: `{path.name}`\n", f"Workloads: {len(wide)}\n"]
    lines.append("\n| variant | wins | ties | losses | median ratio | min ratio | max ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    per_variant = {}
    for v in tested:
        if v not in wide:
            lines.append(f"| {v} | 0 | 0 | 0 | (no data) | (no data) | (no data) |")
            continue
        d = wide.dropna(subset=["fused", v])
        if d.empty:
            lines.append(f"| {v} | 0 | 0 | 0 | (no data) | (no data) | (no data) |")
            continue
        ratio = d.fused / d[v]  # >1 means variant faster than fused
        per_variant[v] = (d, ratio)
        wins = int((ratio > 1 / MARGIN).sum())
        losses = int((ratio < MARGIN).sum())
        ties = len(ratio) - wins - losses
        lines.append(f"| {v} | {wins} | {ties} | {losses} | {ratio.median():.3f}x | {ratio.min():.3f}x | {ratio.max():.3f}x |")

    lines.append("\n## By workload set\n")
    lines.append("\n| variant | set | n | wins | ties | losses | median ratio |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for v in tested:
        if v not in per_variant:
            continue
        d, ratio = per_variant[v]
        d = d.assign(ratio=ratio)
        for s, g in d.groupby("set"):
            r = g.ratio
            wins = int((r > 1 / MARGIN).sum())
            losses = int((r < MARGIN).sum())
            ties = len(r) - wins - losses
            lines.append(f"| {v} | {s} | {len(r)} | {wins} | {ties} | {losses} | {r.median():.3f}x |")

    lines.append("\n## Every rowsize and rife workload\n")
    sel = wide[(wide.set == "rowsize") | (wide.name.str.startswith("rife"))].sort_values(["set", "name", "scale"])
    cols = ["name", "set", "B", "H", "W", "C", "scale", "dtype", "fused", *[v for v in tested if v in wide]]
    cols = [c for c in cols if c in sel]
    lines.append("\n| " + " | ".join(cols) + " |")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
    for _, r in sel.iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) and c not in ("scale",) else str(v))
        lines.append("| " + " | ".join(vals) + " |")

    (path.parent / "improve_summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {path.parent / 'improve_summary.md'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--max-mb", type=int, default=256)
    args = ap.parse_args()

    mx.set_cache_limit(1 << 30)

    out = RESULTS / ("scratch/improve.jsonl" if args.smoke else "improve.jsonl")
    if not args.analyze:
        ws = pick_workloads(args.max_mb)
        print(f"total workloads: {len(ws)}")
        ws = ws[:3] if args.smoke else ws
        store = ResultStore(out)
        write_meta(out.parent / "improve_meta.json", vars(args), list(VARIANTS), SEED)
        run(store, ws)
    analyze(out)


if __name__ == "__main__":
    sys.exit(main())
