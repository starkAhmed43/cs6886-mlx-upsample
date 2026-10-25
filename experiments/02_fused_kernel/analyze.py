"""Speedup table, heatmaps and summary for the fused kernel, forward pass only."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RESULTS = Path(__file__).parent / "results"
WORKLOAD = ["name", "set", "B", "H", "W", "C", "scale", "align_corners", "dtype", "pass"]
UNIQUE = [*WORKLOAD[:-1], "source", "variant", "pass"]
NON_FUSED = ["gather", "gather_compiled", "matmul", "pr4418"]
MARGIN = 0.9  # a method wins only if it is below 0.9 x the other; see 01_crossover for the noise measurement this is based on


def md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(none)\n"
    rows = [[f"{v:.3g}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    lines = [list(map(str, df.columns))] + rows
    fmt = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([fmt(lines[0]), fmt(["---"] * len(lines[0])), *map(fmt, lines[1:])]) + "\n"


def check_unique(df: pd.DataFrame) -> None:
    dup = df[df.duplicated(UNIQUE, keep=False)]
    if len(dup):
        raise ValueError(f"{len(dup)} duplicate rows per workload, variant and pass:\n{dup[UNIQUE].head()}")


def classify(fused_ms: pd.Series, other_ms: pd.Series) -> pd.Series:
    """other_ms / fused_ms. 'fused' wins above 1/MARGIN, 'other' wins below MARGIN."""
    r = other_ms / fused_ms
    return pd.Series(np.where(r > 1 / MARGIN, "fused", np.where(r < MARGIN, "other", "tie")), index=r.index)


def speedup_table(wide: pd.DataFrame) -> pd.DataFrame:
    d = wide.dropna(subset=["fused"]).copy()
    d["best_other"] = d[NON_FUSED].min(axis=1)
    for v in NON_FUSED:
        d[f"speedup_vs_{v}"] = d[v] / d.fused
        present = d[v].notna()
        d[f"win_vs_{v}"] = pd.Series(np.nan, index=d.index, dtype=object)
        d.loc[present, f"win_vs_{v}"] = classify(d.fused[present], d[v][present])
    # best_other is only defined where at least one baseline is present
    has_other = d[NON_FUSED].notna().any(axis=1)
    d["speedup_vs_best_other"] = d.best_other / d.fused
    d["win_vs_best_other"] = pd.Series(np.nan, index=d.index, dtype=object)
    d.loc[has_other, "win_vs_best_other"] = classify(d.fused[has_other], d.best_other[has_other])
    cols = [*WORKLOAD, "fused", *NON_FUSED, "best_other",
            *(f"speedup_vs_{v}" for v in [*NON_FUSED, "best_other"]),
            *(f"win_vs_{v}" for v in [*NON_FUSED, "best_other"])]
    return d[cols].sort_values("speedup_vs_best_other", ascending=False)


def heatmaps(g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.dropna(subset=["fused", *NON_FUSED]).assign(
        best_other=lambda d: d[NON_FUSED].min(axis=1),
        l2=lambda d: np.log2(d[NON_FUSED].min(axis=1) / d.fused),
    )
    hs = sorted(g.H.unique())
    for dtype, grp in g.groupby("dtype"):
        cs = sorted(grp.C.unique())
        fig, axes = plt.subplots(1, len(cs), figsize=(5.5 * len(cs), 4), squeeze=False)
        lim = max(1.0, np.abs(grp.l2).max())
        im = None
        for ax, c in zip(axes[0], cs):
            t = grp[grp.C == c].pivot(index="scale", columns="H", values="l2").reindex(columns=hs)
            im = ax.imshow(t.to_numpy(), cmap="RdBu", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto", origin="lower")
            for (i, j), v in np.ndenumerate(t.to_numpy()):
                if not np.isnan(v):
                    ax.text(j, i, f"{2 ** v:.1f}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(hs)), hs, rotation=90)
            ax.set_yticks(range(len(t)), t.index)
            ax.set_title(f"C={c}")
            ax.set_xlabel("H = W")
        axes[0, 0].set_ylabel("scale")
        fig.colorbar(im, ax=axes[0], label="log2(best of non-fused / fused)  (blue: fused faster, red: fused slower)")
        fig.suptitle(f"forward, {dtype}")
        fig.savefig(fig_dir / f"heatmap_fwd_{dtype}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def real_workloads(r: pd.DataFrame, fig_dir: Path) -> None:
    r = r.dropna(subset=["gather"])
    if r.empty:
        return
    labels = r.assign(label=lambda d: d.name + " B=" + d.B.astype(str) + " " + d.dtype.str.replace("float", "f"))
    labels = labels.sort_values(["name", "B"])
    names = list(dict.fromkeys(labels.label))
    s = labels.set_index("label").reindex(names)
    variants = [v for v in [*NON_FUSED, "fused"] if v in s]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 2))
    for k, v in enumerate(variants):
        ax.barh(np.arange(len(names)) + (k - (len(variants) - 1) / 2) * 0.15, s[v] / s.gather, height=0.15, label=v)
    ax.axvline(1, color="k", lw=0.8)
    ax.set(xscale="log", title="forward", xlabel="time / gather time (below 1: faster)")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "real_workloads.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=RESULTS / "sweep.jsonl")
    args = ap.parse_args()
    out = args.inp.parent
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    df = pd.read_json(args.inp, lines=True, dtype=False)
    check_unique(df)
    valid = df[df.valid == True]  # noqa: E712
    n_bad = len(df) - len(valid)
    print(f"{len(df)} rows, dropped {n_bad} invalid")
    wide = valid.pivot_table(index=WORKLOAD, columns="variant", values="ms_median").reset_index()
    for v in [*NON_FUSED, "fused"]:
        if v not in wide:
            wide[v] = np.nan
    grid_w, real_w = wide[wide.set == "grid"], wide[wide.set == "real"]

    sp = speedup_table(wide)
    sp.to_csv(out / "speedup.csv", index=False)
    heatmaps(grid_w, fig_dir)
    real_workloads(real_w, fig_dir)

    bad = df[df.valid != True].copy()  # noqa: E712
    err = bad["error"].fillna("") if "error" in bad else pd.Series("", index=bad.index)
    bad["error"] = np.where(err.str.contains("OutOfMemory|Insufficient Memory"), "OOM", np.where(err == "", "failed check", err.str[:60]))
    bad = bad.groupby(["name", "dtype", "variant", "error"]).size().reset_index(name="rows")

    def win_counts(col: str) -> dict:
        return sp[col].value_counts().to_dict()

    def excluded(v: str) -> int:
        return int(sp[v].isna().sum()) if v != "best_other" else int(sp[NON_FUSED].isna().all(axis=1).sum())

    def speed_stats(d: pd.DataFrame, col: str) -> str:
        if d.empty:
            return "(no data)"
        return f"median {d[col].median():.2f}x, max {d[col].max():.2f}x"

    up, down = sp[sp.scale >= 1], sp[sp.scale < 1]
    f32, f16 = sp[sp.dtype == "float32"], sp[sp.dtype == "float16"]
    worst = sp.sort_values("speedup_vs_best_other").head(10)[
        [*WORKLOAD, "fused", "best_other", "speedup_vs_best_other", "win_vs_best_other"]]
    real_table = sp[sp.set == "real"][[*WORKLOAD, "fused", *NON_FUSED, "speedup_vs_best_other"]]

    (out / "timing_summary.md").write_text(
        f"# Summary\n\nSource: `{args.inp.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped)\n"
        f"- Grid configs: {len(grid_w)}; real configs: {len(real_w)}\n"
        "- A method wins only if it is below 0.9 x the other time; otherwise the pair counts as a tie "
        "(same 10% noise margin as 01_crossover).\n"
        "- A workload is excluded from a baseline's win/tie/loss counts (and from best_other) when "
        "that baseline has no valid row for it (e.g. invalid/OOM).\n\n"
        "## Wins (fused vs each baseline)\n\n" +
        "".join(f"- vs {v}: {win_counts(f'win_vs_{v}')} "
                f"({excluded(v)} excluded, {'no baseline data at all' if v == 'best_other' else f'no {v} data'})\n"
                for v in [*NON_FUSED, "best_other"]) +
        "\n## Speedup (fused vs best of the non-fused variants)\n\n"
        f"- Overall: {speed_stats(sp, 'speedup_vs_best_other')}\n"
        f"- Upscale (scale >= 1): {speed_stats(up, 'speedup_vs_best_other')}\n"
        f"- Downscale (scale < 1): {speed_stats(down, 'speedup_vs_best_other')}\n"
        f"- float32: {speed_stats(f32, 'speedup_vs_best_other')}\n"
        f"- float16: {speed_stats(f16, 'speedup_vs_best_other')}\n\n"
        f"## Invalid rows\n\n{md(bad)}\n"
        f"## Real workloads\n\n{md(real_table)}\n"
        f"## Worst 10 cases for fused (vs best of the non-fused variants)\n\n{md(worst)}\n"
    )
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
