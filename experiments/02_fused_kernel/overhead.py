"""Does a fixed per-call cost dominate `fused` on small workloads, and does
rebuilding the index/weight tables explain it?

Times `fused`, `fused_cached` (tables built once and reused via `lru_cache`),
`tables_only` (builds the tables but never runs the kernel), a near-zero
kernel-launch `floor`, and each workload's best non-fused variant (from
`results/speedup.csv`, the 01_crossover timing sweep already run for this
kernel). See `experiments/02_fused_kernel/README.md`.
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
import pandas as pd

from mlx_upsample.bench import ResultStore, maxdiff, make_fwd, row_key, time_interleaved, write_meta
from mlx_upsample.kernels.bilinear import axis_tables, fused, fused_uncached
from mlx_upsample.variants import VARIANTS as STOCK_VARIANTS
from mlx_upsample.workloads import grid, make_input, real

RESULTS = Path(__file__).parent / "results"
SPEEDUP_CSV = RESULTS / "speedup.csv"
SEED = 0
MARGIN = 0.9  # same 10% noise margin as speedup.csv
NON_FUSED = ["gather", "gather_compiled", "matmul", "pr4418"]
IDENT = ["name", "set", "B", "H", "W", "C", "scale", "align_corners", "dtype"]
FIXED = ["fused", "fused_cached", "tables_only", "floor"]
SMOKE_N = 5


def _key(d) -> tuple:
    """Identity tuple for a workload, usable on a Workload (via asdict) or a speedup.csv row."""
    return (
        str(d["name"]), str(d["set"]), int(d["B"]), int(d["H"]), int(d["W"]), int(d["C"]),
        round(float(d["scale"]), 9), bool(d["align_corners"]), str(d["dtype"]),
    )


def pick_workloads(catalog_by_key: dict, sp: pd.DataFrame) -> list:
    """(a) ties/losses for fused, (b) every real workload, (c) a size ladder; deduplicated."""
    selected = {}
    for _, row in sp[sp.win_vs_best_other != "fused"].iterrows():
        k = _key(row)
        if k in catalog_by_key:
            selected[k] = catalog_by_key[k]
    for w in real():
        selected[_key(asdict(w))] = w
    for w in grid():
        if w.C == 4 and w.dtype == "float32" and w.scale in (2, 0.25):
            selected[_key(asdict(w))] = w
    return [selected[k] for k in sorted(selected)]


def make_tables_only(w) -> callable:
    sf = (w.scale, w.scale)

    def run():
        yl, yr, yw = axis_tables(w.H, sf[0], w.align_corners)
        xl, xr, xw = axis_tables(w.W, sf[1], w.align_corners)
        mx.eval(yl, yr, yw, xl, xr, xw)
        return yl, yr, yw, xl, xr, xw

    return run


def make_floor() -> callable:
    # The cheapest possible GPU op: one scalar add. Its time is a floor on
    # how fast any `mx.eval`'d call can be, no matter what it computes.
    x = mx.array([0.0], dtype=mx.float32)

    def run():
        y = x + 1
        mx.eval(y)
        return y

    return run


def process_workload(i: int, w, sp_row: pd.Series, store: ResultStore) -> None:
    sf = (w.scale, w.scale)
    x = make_input(w)
    y_ref = STOCK_VARIANTS["gather"](x, sf, w.align_corners)
    mx.eval(y_ref)
    y_ref = y_ref.astype(mx.float32)
    atol = 1e-5 if w.dtype == "float32" else 1e-2

    non_fused = {v: sp_row[v] for v in NON_FUSED if pd.notna(sp_row[v])}
    best_other_name = min(non_fused, key=non_fused.get) if non_fused else None

    variants = {"fused": fused_uncached, "fused_cached": fused}
    if best_other_name:
        variants[best_other_name] = STOCK_VARIANTS[best_other_name]

    todo = [v for v in [*variants, "tables_only", "floor"]
            if not store.has(row_key({**asdict(w), "variant": v, "pass": "fwd"}))]
    if not todo:
        return

    # Warm the fused_cached (now `fused`) table cache before it is ever timed.
    if "fused_cached" in todo:
        mx.eval(fused(x, sf, w.align_corners))

    fns, valid = {}, {}
    for v in todo:
        if v == "tables_only":
            fns[v], valid[v] = make_tables_only(w), True
        elif v == "floor":
            fns[v], valid[v] = make_floor(), True
        else:
            f = variants[v]
            y = f(x, sf, w.align_corners)
            mx.eval(y)
            valid[v] = bool(maxdiff(y, y_ref) <= atol)
            fns[v] = make_fwd(f, x, sf, w.align_corners)

    times = time_interleaved(fns, seed=i)
    for v in todo:
        ms = times[v]
        store.append({
            **asdict(w), "variant": v, "pass": "fwd",
            "ms_median": sorted(ms)[len(ms) // 2], "ms_repeats": ms,
            "valid": valid[v], "best_other_name": best_other_name,
        })


def classify(fast_ms: pd.Series, other_ms: pd.Series) -> pd.Series:
    r = other_ms / fast_ms
    return pd.Series(np.where(r > 1 / MARGIN, "fused", np.where(r < MARGIN, "other", "tie")), index=r.index)


def analyze(store_path: Path, out_dir: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_json(store_path, lines=True, dtype=False)
    valid = df[df.valid == True]  # noqa: E712
    n_bad = len(df) - len(valid)
    print(f"{len(df)} rows, dropped {n_bad} invalid")

    wide = valid.pivot_table(index=IDENT, columns="variant", values="ms_median", aggfunc="first").reset_index()
    bo_name = valid.groupby(IDENT)["best_other_name"].first().reset_index()
    wide = wide.merge(bo_name, on=IDENT, how="left")
    for v in [*FIXED]:
        if v not in wide:
            wide[v] = np.nan

    def best_other_ms(r):
        return r[r.best_other_name] if pd.notna(r.best_other_name) and r.best_other_name in r else np.nan

    wide["best_other_ms"] = wide.apply(best_other_ms, axis=1)
    wide["out_elements"] = wide.apply(
        lambda r: int(r.B) * int(r.scale * r.H) * int(r.scale * r.W) * int(r.C), axis=1
    )
    wide["saved_ms"] = wide.fused - wide.fused_cached
    # `tables_only` is itself timed as its own `mx.eval` call, so it includes the same
    # per-call launch floor as everything else. Subtract `floor` to get the net cost of
    # building the tables, which is the fair comparison against `saved_ms`.
    wide["tables_net_ms"] = wide.tables_only - wide.floor
    wide["tables_only_share_of_fused"] = wide.tables_only / wide.fused
    wide["speedup_cached_vs_best_other"] = wide.best_other_ms / wide.fused_cached

    has_other = wide.best_other_ms.notna()
    wide["win_vs_best_other"] = pd.Series(np.nan, index=wide.index, dtype=object)
    wide.loc[has_other, "win_vs_best_other"] = classify(wide.fused[has_other], wide.best_other_ms[has_other])
    wide["win_vs_best_other_cached"] = pd.Series(np.nan, index=wide.index, dtype=object)
    wide.loc[has_other, "win_vs_best_other_cached"] = classify(
        wide.fused_cached[has_other], wide.best_other_ms[has_other]
    )

    cols = [*IDENT, *FIXED, "tables_net_ms", "best_other_name", "best_other_ms", "out_elements",
            "saved_ms", "tables_only_share_of_fused", "speedup_cached_vs_best_other",
            "win_vs_best_other", "win_vs_best_other_cached"]
    wide[cols].to_csv(out_dir / "overhead.csv", index=False)

    became_wins = wide[has_other & (wide.win_vs_best_other != "fused") & (wide.win_vs_best_other_cached == "fused")]

    # Match this rerun against the sweep (`speedup.csv`) by workload key, not just by count,
    # since noise at these sizes moves workloads in both directions (sweep win -> rerun non-win,
    # and sweep non-win -> rerun win).
    sp = pd.read_csv(SPEEDUP_CSV)
    wide_keys = wide.apply(_key, axis=1)
    sp_non_win_keys = set(sp[sp.win_vs_best_other != "fused"].apply(_key, axis=1))
    sp_win_keys = set(sp[sp.win_vs_best_other == "fused"].apply(_key, axis=1))
    sp_non_wins_n = len(sp_non_win_keys)
    overlap_mask = wide_keys.isin(sp_non_win_keys)
    still_non_win_n = int((overlap_mask & (wide.win_vs_best_other != "fused")).sum())
    flipped_to_win_n = int((overlap_mask & (wide.win_vs_best_other == "fused")).sum())
    flipped_to_non_win_n = int((wide_keys.isin(sp_win_keys) & (wide.win_vs_best_other != "fused")).sum())
    total_non_win_n = still_non_win_n + flipped_to_non_win_n

    median_saved = wide.saved_ms.median()
    median_tables_net = wide.tables_net_ms.median()
    median_floor = wide.floor.median()
    close = abs(median_saved - median_tables_net) <= 0.25 * max(median_saved, median_tables_net, 1e-9)

    small = wide[wide.fused < 0.7]
    small_saved = small.saved_ms.median()
    small_tables_net = small.tables_net_ms.median()
    small_close = abs(small_saved - small_tables_net) <= 0.25 * max(small_saved, small_tables_net, 1e-9)

    wins_fused = int((wide.win_vs_best_other == "fused").sum())
    wins_cached = int((wide.win_vs_best_other_cached == "fused").sum())

    (out_dir / "overhead_summary.md").write_text(
        f"# Fixed-cost summary\n\nSource: `{store_path.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped); workloads analyzed: {len(wide)}\n"
        f"- `tables_only` is itself an `mx.eval` call, so it includes the per-call launch "
        f"`floor`. `tables_net_ms` = `tables_only` - `floor` is the net table cost, and is "
        f"what `saved_ms` (`fused` - `fused_cached`) should be compared against.\n"
        f"- Median saved time vs median net table time (all {len(wide)} workloads): "
        f"{median_saved:.4f} ms vs {median_tables_net:.4f} ms "
        f"(within 25%: {close})\n"
        f"- Same comparison restricted to `fused` < 0.7 ms ({len(small)} workloads): "
        f"{small_saved:.4f} ms vs {small_tables_net:.4f} ms (within 25%: {small_close})\n"
        f"- Median launch `floor`: {median_floor:.4f} ms\n"
        f"- `fused` wins vs best other: {wins_fused} of {len(wide[has_other])}; "
        f"`fused_cached` wins: {wins_cached} of {len(wide[has_other])}\n"
        f"- Of the {sp_non_wins_n} sweep (`speedup.csv`) ties/losses for uncached `fused`, "
        f"{still_non_win_n} are still ties or losses for uncached `fused` in this rerun, and "
        f"{flipped_to_win_n} flipped to wins with no code change, and {flipped_to_non_win_n} "
        f"workload(s) that won in the sweep flipped the other way to a tie or loss here. "
        f"Total non-wins in this rerun: {total_non_win_n}. The noise at these sub-ms sizes "
        f"goes both ways.\n"
        f"- Ties/losses for uncached `fused` in this rerun that become wins once tables are "
        f"cached: {len(became_wins)} of {total_non_win_n}\n"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"fused": "C0", "fused_cached": "C1", "tables_only": "C2", "floor": "C3", "best_other": "C4"}
    for v, label in [("fused", "fused"), ("fused_cached", "fused_cached"),
                     ("tables_only", "tables_only"), ("floor", "floor")]:
        d = wide.dropna(subset=[v]).sort_values("out_elements")
        ax.plot(d.out_elements, d[v], "o-", ms=3, lw=1, label=label, color=colors[v])
    d = wide.dropna(subset=["best_other_ms"]).sort_values("out_elements")
    ax.plot(d.out_elements, d.best_other_ms, "o-", ms=3, lw=1, label="best_other", color=colors["best_other"])
    ax.set(xscale="log", yscale="log", xlabel="output elements", ylabel="time (ms)",
           title="Fixed-cost check: forward time vs output size")
    ax.legend()
    fig.savefig(fig_dir / "overhead.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote outputs to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()

    out_dir = RESULTS / "scratch" if args.smoke else RESULTS
    out_dir.mkdir(parents=True, exist_ok=True)
    store_path = out_dir / "overhead.jsonl"

    if args.analyze:
        analyze(store_path, out_dir)
        return

    sp = pd.read_csv(SPEEDUP_CSV)
    catalog_by_key = {_key(asdict(w)): w for w in grid() + real()}
    ws = pick_workloads(catalog_by_key, sp)
    if args.smoke:
        ws = ws[:SMOKE_N]

    store = ResultStore(store_path)
    write_meta(out_dir / "overhead_meta.json", vars(args), [*FIXED, "best_other"], SEED)

    sp_by_key = {_key(row): row for _, row in sp.iterrows()}
    for i, w in enumerate(ws):
        row = sp_by_key.get(_key(asdict(w)))
        if row is None:
            print(f"skip {w.name}: not in {SPEEDUP_CSV.name}", flush=True)
            continue
        process_workload(i, w, row, store)
        print(f"{i + 1}/{len(ws)} {w.name} B={w.B} {w.dtype} done", flush=True)
        mx.clear_cache()

    analyze(store_path, out_dir)


if __name__ == "__main__":
    sys.exit(main())
