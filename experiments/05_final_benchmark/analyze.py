"""Median-over-runs speedup of fused_cpp vs stock nn.Upsample and vs pr4418, on the real catalog.

Speedup = other_ms / fused_cpp_ms (median over runs). Values above 1 mean fused_cpp is faster.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).parent / "results"
WORKLOAD = ["name", "set", "B", "H", "W", "C", "scale", "align_corners", "dtype", "pass"]
UNIQUE = [*WORKLOAD[:-1], "source", "variant", "pass", "run"]
VARIANTS = ["stock", "pr4418", "fused_cpp"]
OTHERS = ["stock", "pr4418"]
MARGIN = 0.9  # a method wins only if it is below 0.9x the other; same rule as 01/02/03/04


def gmean(x: pd.Series) -> float:
    x = np.asarray(x.dropna(), dtype=float)
    return float(np.exp(np.mean(np.log(x)))) if len(x) else float("nan")


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
        raise ValueError(f"{len(dup)} duplicate rows per workload, variant, pass and run:\n{dup[UNIQUE].head()}")


def classify(ratio: pd.Series) -> pd.Series:
    """ratio = other_ms / fused_cpp_ms. 'fused_cpp' wins above 1/MARGIN, 'other' wins below MARGIN.
    NaN ratio (baseline missing) is its own 'missing' bucket, not a tie."""
    return pd.Series(
        np.where(ratio.isna(), "missing", np.where(ratio > 1 / MARGIN, "fused_cpp", np.where(ratio < MARGIN, "other", "tie"))),
        index=ratio.index,
    )


def speedup_table(med: pd.DataFrame) -> pd.DataFrame:
    wide = med.pivot_table(index=WORKLOAD, columns="variant", values="ms_median").reset_index()
    for v in VARIANTS:
        if v not in wide:
            wide[v] = np.nan
    d = wide.dropna(subset=["fused_cpp"]).copy()
    for other in OTHERS:
        d[f"speedup_vs_{other}"] = d[other] / d.fused_cpp
        d[f"win_vs_{other}"] = classify(d[f"speedup_vs_{other}"])
    cols = [*WORKLOAD, *VARIANTS, *[f"speedup_vs_{o}" for o in OTHERS], *[f"win_vs_{o}" for o in OTHERS]]
    return d[cols].sort_values("speedup_vs_stock", ascending=False)


def figure(sp: pd.DataFrame, path: Path) -> None:
    labels = sp.assign(label=lambda d: d.name + " B=" + d.B.astype(str) + " " + d.dtype.str.replace("float", "f"))
    order = list(dict.fromkeys(labels.sort_values(["name", "B"]).label))
    s = labels.set_index(["label", "pass"]).speedup_vs_stock
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(order) + 2))
    for k, pas in enumerate(("fwd", "fwd_bwd")):
        vals = [s.get((lab, pas), np.nan) for lab in order]
        ax.barh(y + (k - 0.5) * 0.3, vals, height=0.3, label=pas)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_xscale("log")
    ax.set_yticks(y, order, fontsize=6)
    ax.invert_yaxis()
    ax.set(xlabel="fused_cpp speedup vs stock nn.Upsample (log scale)", title="05 final benchmark: real-model catalog")
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    inp = args.inp or (RESULTS / "scratch/runs.jsonl" if args.smoke else RESULTS / "runs.jsonl")
    out = args.out_dir or inp.parent
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_json(inp, lines=True, dtype=False)
    check_unique(df)
    valid = df[df.valid == True]  # noqa: E712
    n_bad = len(df) - len(valid)
    print(f"{len(df)} rows, dropped {n_bad} invalid")

    g = valid.groupby([*WORKLOAD, "variant"]).ms_median
    med = g.median().reset_index()
    spread = g.agg(lambda s: (s.max() - s.min()) / s.median() if s.median() else np.nan).reset_index(name="spread")
    med = med.merge(spread, on=[*WORKLOAD, "variant"])
    if "grad_rel_err" in valid:
        grad_max = valid.groupby([*WORKLOAD, "variant"]).grad_rel_err.max().reset_index(name="grad_rel_err_max")
        med = med.merge(grad_max, on=[*WORKLOAD, "variant"], how="left")
    med.to_csv(out / "table.csv", index=False)

    sp = speedup_table(med)
    figure(sp, fig_dir / "speedup.png")

    geo, counts = {}, {}
    for pas, grp in sp.groupby("pass"):
        geo[pas] = {o: gmean(grp[f"speedup_vs_{o}"]) for o in OTHERS}
        counts[pas] = {o: grp[f"win_vs_{o}"].value_counts().to_dict() for o in OTHERS}

    bad = df[df.valid != True].copy()  # noqa: E712
    err = bad["error"].fillna("") if "error" in bad else pd.Series("", index=bad.index)
    bad["error"] = np.where(err.str.contains("OutOfMemory|Insufficient Memory"), "OOM", np.where(err == "", "failed check", err.str[:60]))
    bad = bad.groupby(["name", "dtype", "env", "variant", "pass", "error"]).size().reset_index(name="rows")

    out_dtype = valid.groupby(["variant", "dtype"]).out_dtype.agg(lambda s: sorted(set(s))).reset_index()

    if "grad_rel_err" in valid:
        grad_err = valid.groupby(["variant", "dtype"]).grad_rel_err.max().reset_index(name="grad_rel_err_max")
    else:
        grad_err = pd.DataFrame(columns=["variant", "dtype", "grad_rel_err_max"])

    def fmt_pass(pas: str) -> str:
        lines = [f"### {pas}\n\n"]
        for o in OTHERS:
            lines.append(f"- vs {o}: geomean speedup {geo[pas][o]:.3f}x, win/tie/loss {counts[pas][o]}\n")
        return "".join(lines) + "\n"

    (out / "summary.md").write_text(
        f"# Summary\n\nSource: `{inp.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped)\n"
        f"- Workloads: {sp.name.nunique()} (of {med.name.nunique()} with any valid rows)\n"
        f"- Each (workload, variant, pass) median is over up to 10 fresh-process runs; `spread` in "
        "`table.csv` is (max - min) / median across those runs.\n"
        "- `speedup_vs_X` = X's median ms / fused_cpp's median ms; above 1 means fused_cpp is faster. "
        "A method wins only if it is below 0.9x the other (fused_cpp wins) or above 1/0.9x (the other "
        "wins); otherwise it is a tie (same 10% noise margin as 01_crossover/02_fused_kernel/03_tiling/04_backward).\n\n"
        "## Geometric mean speedup and win/tie/loss, fused_cpp vs stock nn.Upsample path\n\n"
        + "".join(fmt_pass(p) for p in ("fwd", "fwd_bwd") if p in geo)
        + f"## Output dtype by variant and input dtype\n\n{md(out_dtype)}\n"
        f"## Max gradient relative error by variant and input dtype (fwd_bwd pass)\n\n{md(grad_err)}\n"
        f"## Invalid rows\n\n{md(bad)}\n"
        f"## Fastest and slowest real workloads (by speedup vs stock)\n\n{md(pd.concat([sp.head(5), sp.tail(5)]))}\n"
    )
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
