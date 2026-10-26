"""Speedup table, heatmaps, gradient table and summary for fused (fwd+bwd) vs stock MLX.

Speedup = other_ms / fused_ms for each stock variant (gather, matmul, pr4418) and vs
best_stock = min(gather, matmul, pr4418). Values above 1 mean fused is faster.
"""

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
OTHERS = ["gather", "matmul", "pr4418"]
MARGIN = 0.9  # a method wins only if it is below 0.9 x the other; see 02_fused_kernel


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


def classify(ratio: pd.Series) -> pd.Series:
    """ratio = other_ms / fused_ms. 'fused' wins above 1/MARGIN, 'other' wins below MARGIN."""
    return pd.Series(np.where(ratio > 1 / MARGIN, "fused", np.where(ratio < MARGIN, "other", "tie")), index=ratio.index)


def speedup_table(wide: pd.DataFrame) -> pd.DataFrame:
    d = wide.dropna(subset=["fused"]).copy()
    d["best_stock"] = d[OTHERS].min(axis=1, skipna=True)
    for v in [*OTHERS, "best_stock"]:
        d[f"speedup_{v}"] = d[v] / d.fused
        d[f"win_{v}"] = classify(d[f"speedup_{v}"])
    cols = [*WORKLOAD, "fused", *OTHERS, "best_stock", *[f"speedup_{v}" for v in [*OTHERS, "best_stock"]], *[f"win_{v}" for v in [*OTHERS, "best_stock"]]]
    return d[cols].sort_values("speedup_best_stock", ascending=False)


def direction(scale: pd.Series) -> pd.Series:
    return pd.Series(np.where(scale > 1, "upscale", np.where(scale < 1, "downscale", "same")), index=scale.index)


def win_summary(sp: pd.DataFrame) -> dict:
    """Win/tie/loss counts and speedup-vs-best_stock stats, split by set, direction and dtype."""
    d = sp.assign(direction=direction(sp.scale))
    out = {"overall": {"counts": sp.win_best_stock.value_counts().to_dict(),
                        "median": sp.speedup_best_stock.median(), "min": sp.speedup_best_stock.min(),
                        "max": sp.speedup_best_stock.max()}}
    for key, col in (("set", "set"), ("direction", "direction"), ("dtype", "dtype")):
        for val, grp in d.groupby(col):
            if grp.empty:
                continue
            out[f"{key}={val}"] = {"counts": grp.win_best_stock.value_counts().to_dict(),
                                    "median": grp.speedup_best_stock.median(), "min": grp.speedup_best_stock.min(),
                                    "max": grp.speedup_best_stock.max()}
    return out


def heatmaps(g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.dropna(subset=["fused", "pr4418", "gather", "matmul"]).copy()
    g["best_stock"] = g[OTHERS].min(axis=1, skipna=True)
    g = g.assign(l2=lambda d: np.log2(d.best_stock / d.fused))
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
                    ax.text(j, i, f"{2 ** v:.2f}", ha="center", va="center", fontsize=6)
            ax.set_xticks(range(len(hs)), hs, rotation=90)
            ax.set_yticks(range(len(t)), t.index)
            ax.set_title(f"C={c}")
            ax.set_xlabel("H = W")
        axes[0, 0].set_ylabel("scale")
        fig.colorbar(im, ax=axes[0], label="log2(best_stock / fused)  (positive/blue: fused faster)")
        fig.suptitle(f"forward+backward, {dtype}")
        fig.savefig(fig_dir / f"heatmap_fwd_bwd_{dtype}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def real_workloads(r: pd.DataFrame, fig_dir: Path) -> None:
    r = r.dropna(subset=["fused"])
    if r.empty:
        return
    labels = r.assign(label=lambda d: d.name + " B=" + d.B.astype(str) + " " + d.dtype.str.replace("float", "f"))
    labels = labels.sort_values(["name", "B"])
    names = list(dict.fromkeys(labels.label))
    s = labels.set_index("label").reindex(names)
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 2))
    for k, v in enumerate(OTHERS):
        ax.barh(np.arange(len(names)) + (k - 1) * 0.27, s[v] / s.fused, height=0.27, label=v)
    ax.axvline(1, color="k", lw=0.8)
    ax.set(xscale="log", title="forward+backward", xlabel="other time / fused time (above 1: fused faster)")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.invert_yaxis()
    ax.legend(fontsize=7)
    fig.savefig(fig_dir / "real_workloads.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def gradient_table(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per (variant, dtype): max/median grad_rel_err against the float32 gather reference, and invalid counts."""
    if "grad_rel_err" not in df:
        return pd.DataFrame(), pd.DataFrame()
    ge = df[df["pass"] == "fwd_bwd"].dropna(subset=["grad_rel_err"])
    gstats = ge.groupby(["variant", "dtype"]).grad_rel_err.agg(median="median", max="max", n="size").reset_index()
    invalid = ge.groupby(["variant", "dtype"]).valid.agg(lambda s: int((~s).sum())).reset_index(name="n_invalid")
    gstats = gstats.merge(invalid, on=["variant", "dtype"])
    return gstats, ge.sort_values("grad_rel_err", ascending=False).head(10)[
        ["name", "B", "scale", "C", "dtype", "variant", "grad_rel_err"]
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=RESULTS / "sweep.jsonl")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    out = args.out_dir or args.inp.parent
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(args.inp, lines=True, dtype=False)
    check_unique(df)
    valid = df[df.valid == True]  # noqa: E712
    n_bad = len(df) - len(valid)
    print(f"{len(df)} rows, dropped {n_bad} invalid")
    wide = valid.pivot_table(index=WORKLOAD, columns="variant", values="ms_median").reset_index()
    for v in [*OTHERS, "fused"]:
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

    excluded = int(wide["fused"].isna().sum())
    gstats, gtop = gradient_table(df)

    ws = win_summary(sp)

    def fmt_group(label: str, d: dict) -> str:
        return f"- {label}: {d['counts']}, speedup vs best_stock median {d['median']:.3f}x, min {d['min']:.3f}x, max {d['max']:.3f}x\n"

    real_table = sp[sp.set == "real"][[*WORKLOAD, "fused", *OTHERS, "best_stock", "speedup_best_stock", "win_best_stock"]]

    (out / "summary.md").write_text(
        f"# Summary\n\nSource: `{args.inp.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped)\n"
        f"- Grid configs: {len(grid_w)}; real configs: {len(real_w)}\n"
        f"- {excluded} workload(s) excluded from the speedup table (missing fused row).\n"
        "- `speedup_X` = X time / fused time; above 1 means fused is faster. A method wins only if it "
        "is below 0.9x the other (fused wins) or above 1/0.9x (the other wins); otherwise it is a tie "
        "(same 10% noise margin as 01_crossover/02_fused_kernel/03_tiling).\n\n"
        "## Wins vs best_stock (min of gather, matmul, pr4418), forward+backward\n\n"
        + fmt_group("overall", ws["overall"])
        + "".join(fmt_group(k, v) for k in ws if k != "overall" for v in [ws[k]])
        + f"\n## Invalid rows\n\n{md(bad)}\n"
        f"## Gradient error against a float32 gather reference\n\n"
        "Relative error is the largest absolute difference divided by the largest absolute reference "
        "gradient. float32 is gated at <= 1e-5 (invalid otherwise); float16 is recorded, not gated.\n\n"
        f"{md(gstats)}\nTen worst rows:\n\n{md(gtop)}\n"
        f"## Real workloads\n\n{md(real_table)}\n"
    )
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
