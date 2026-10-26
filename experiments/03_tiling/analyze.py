"""Speedup table, heatmaps and summary for fused_tiled vs fused, forward pass only.

ratio = fused_ms / fused_tiled_ms: greater than 1 means tiling is faster.
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
MARGIN = 0.9  # a method wins only if it is below 0.9 x the other; see 02_fused_kernel
DTYPE_BYTES = {"float32": 4, "float16": 2}
BW_BUCKETS = [(0, 10), (10, 50), (50, 100), (100, 300)]  # min_bytes, in MB (1e6 bytes)


def add_bandwidth(wide: pd.DataFrame) -> pd.DataFrame:
    """min_bytes = input bytes (input dtype) + output bytes (float32); gbs_* = min_bytes / ms / 1e6.

    min_bytes counts each input byte once, so it is a lower bound on real DRAM traffic.
    """
    wide = wide.copy()
    oh, ow = (wide.scale * wide.H).astype(int), (wide.scale * wide.W).astype(int)
    in_bytes = wide.B * wide.H * wide.W * wide.C * wide.dtype.map(DTYPE_BYTES)
    out_bytes = wide.B * oh * ow * wide.C * 4
    wide["min_bytes"] = in_bytes + out_bytes
    for v in ("fused", "fused_tiled"):
        wide[f"gbs_{v}"] = wide.min_bytes / wide[v] / 1e6
    return wide


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
    """ratio = fused_ms / fused_tiled_ms. 'tiled' wins above 1/MARGIN, 'fused' wins below MARGIN."""
    return pd.Series(np.where(ratio > 1 / MARGIN, "tiled", np.where(ratio < MARGIN, "fused", "tie")), index=ratio.index)


def speedup_table(wide: pd.DataFrame) -> pd.DataFrame:
    d = wide.dropna(subset=["fused", "fused_tiled"]).copy()
    d["ratio"] = d.fused / d.fused_tiled
    d["win"] = classify(d.ratio)
    cols = [*WORKLOAD, "fused", "fused_tiled", "ratio", "win", "min_bytes", "gbs_fused", "gbs_fused_tiled"]
    return d[cols].sort_values("ratio", ascending=False)


def bandwidth_table(sp: pd.DataFrame) -> pd.DataFrame:
    """Median gbs_fused, gbs_fused_tiled and their ratio, bucketed by min_bytes (MB, 1e6 bytes)."""
    rows = []
    for lo, hi in BW_BUCKETS:
        d = sp[(sp.min_bytes / 1e6 > lo) & (sp.min_bytes / 1e6 <= hi)] if lo else sp[sp.min_bytes / 1e6 <= hi]
        label = f"<= {hi} MB" if lo == 0 else f"{lo}-{hi} MB"
        if d.empty:
            rows.append({"bucket": label, "n": 0, "gbs_fused": np.nan, "gbs_fused_tiled": np.nan, "ratio": np.nan})
            continue
        gf, gt = d.gbs_fused.median(), d.gbs_fused_tiled.median()
        rows.append({"bucket": label, "n": len(d), "gbs_fused": gf, "gbs_fused_tiled": gt, "ratio": gf / gt})
    return pd.DataFrame(rows)


def heatmaps(g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.dropna(subset=["fused", "fused_tiled"]).assign(l2=lambda d: np.log2(d.fused / d.fused_tiled))
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
        fig.colorbar(im, ax=axes[0], label="log2(fused / fused_tiled)  (red: tiling slower, blue: tiling faster)")
        fig.suptitle(f"forward, {dtype}")
        fig.savefig(fig_dir / f"heatmap_fwd_{dtype}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def real_workloads(r: pd.DataFrame, fig_dir: Path) -> None:
    r = r.dropna(subset=["fused", "fused_tiled"])
    if r.empty:
        return
    labels = r.assign(label=lambda d: d.name + " B=" + d.B.astype(str) + " " + d.dtype.str.replace("float", "f"))
    labels = labels.sort_values(["name", "B"])
    names = list(dict.fromkeys(labels.label))
    s = labels.set_index("label").reindex(names)
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(names) + 2))
    ax.barh(np.arange(len(names)), s.fused / s.fused_tiled, height=0.5)
    ax.axvline(1, color="k", lw=0.8)
    ax.set(xscale="log", title="forward", xlabel="fused time / fused_tiled time (above 1: tiling faster)")
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.invert_yaxis()
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
    for v in ("fused", "fused_tiled"):
        if v not in wide:
            wide[v] = np.nan
    wide = add_bandwidth(wide)
    grid_w, real_w = wide[wide.set == "grid"], wide[wide.set == "real"]

    sp = speedup_table(wide)
    sp.to_csv(out / "speedup.csv", index=False)
    heatmaps(grid_w, fig_dir)
    real_workloads(real_w, fig_dir)

    bad = df[df.valid != True].copy()  # noqa: E712
    err = bad["error"].fillna("") if "error" in bad else pd.Series("", index=bad.index)
    bad["error"] = np.where(err.str.contains("OutOfMemory|Insufficient Memory"), "OOM", np.where(err == "", "failed check", err.str[:60]))
    bad = bad.groupby(["name", "dtype", "variant", "error"]).size().reset_index(name="rows")

    excluded = int(wide[["fused", "fused_tiled"]].isna().any(axis=1).sum())

    def win_counts() -> dict:
        return sp["win"].value_counts().to_dict()

    def ratio_stats(d: pd.DataFrame) -> str:
        if d.empty:
            return "(no data)"
        return f"median {d.ratio.median():.3f}x, min {d.ratio.min():.3f}x, max {d.ratio.max():.3f}x"

    up = sp  # sweep already filters to scale > 1; kept as a named view for clarity
    f32, f16 = sp[sp.dtype == "float32"], sp[sp.dtype == "float16"]
    c_groups = {c: sp[sp.C == c] for c in sorted(sp.C.unique())}
    real_table = sp[sp.set == "real"][[*WORKLOAD, "fused", "fused_tiled", "ratio", "win"]]
    bw = bandwidth_table(sp)

    (out / "timing_summary.md").write_text(
        f"# Summary\n\nSource: `{args.inp.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped)\n"
        f"- Grid configs: {len(grid_w)}; real configs: {len(real_w)}\n"
        f"- {excluded} workload(s) excluded from the speedup table (missing fused or fused_tiled row).\n"
        "- A method wins only if it is below 0.9x the other time; otherwise the pair counts as a tie "
        "(same 10% noise margin as 02_fused_kernel). ratio = fused_ms / fused_tiled_ms.\n\n"
        f"## Wins (fused_tiled vs fused)\n\n{win_counts()}\n\n"
        "## Ratio (fused / fused_tiled; above 1 means tiling is faster)\n\n"
        f"- Overall: {ratio_stats(up)}\n"
        f"- float32: {ratio_stats(f32)}\n"
        f"- float16: {ratio_stats(f16)}\n" +
        "".join(f"- C={c}: {ratio_stats(d)}\n" for c, d in c_groups.items()) +
        "\n## Effective bandwidth (GB/s), by min_bytes bucket\n\n"
        "min_bytes = input bytes (input dtype) + output bytes (output is always float32); "
        "it counts each input byte once, so it is a lower bound on real DRAM traffic.\n\n"
        f"{md(bw)}\n"
        f"\n## Invalid rows\n\n{md(bad)}\n"
        f"## Real workloads\n\n{md(real_table)}\n"
    )
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
