"""Crossover table, heatmaps, gate regret and summary from sweep.jsonl."""

import argparse
import json
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
OTHERS = ["gather_compiled", "matmul", "pr4418"]
MARGIN = 0.9  # a method wins only if it is below 0.9 x the other; identical code paths differ p50 1.2%, p90 ~8%, p95 ~15%, max 59%
SLOW = 1.1  # the gate is wrong only if its choice is more than 10% slower than the best
PR_MAX_INPUT = 128  # PR #4418 uses matmul only for scale >= 1 and H, W <= 128


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


def crossover(g: pd.DataFrame) -> pd.DataFrame:
    """Per (pass, scale, C, dtype): sizes where matmul wins, ties, and the largest size present."""
    out = []
    for (p, s, c, d), grp in g.dropna(subset=["gather", "matmul"]).groupby(["pass", "scale", "C", "dtype"]):
        grp = grp.sort_values("H")
        win = (grp.matmul < MARGIN * grp.gather).to_numpy()
        lose = (grp.gather < MARGIN * grp.matmul).to_numpy()
        hs = grp.H.to_numpy()
        idx = np.flatnonzero(win)
        out.append({
            "pass": p, "scale": s, "C": c, "dtype": d,
            "first_win_H": hs[idx[0]] if len(idx) else np.nan,
            "last_win_H": hs[idx[-1]] if len(idx) else np.nan,
            "n_wins": len(idx), "n_ties": int((~win & ~lose).sum()), "n_sizes": len(hs), "max_H": hs.max(),
            "status": "always" if win.all() else "never" if not win.any() else "partial",
            # no size where gather clearly wins lies between the first and the last matmul win
            "contiguous": bool(len(idx) == 0 or not lose[idx[0] : idx[-1] + 1].any()),
        })
    return pd.DataFrame(out)


def heatmaps(g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.dropna(subset=["gather", "matmul"]).assign(l2=lambda d: np.log2(d.gather / d.matmul))
    hs = sorted(g.H.unique())
    for (p, d), grp in g.groupby(["pass", "dtype"]):
        cs = sorted(grp.C.unique())
        fig, axes = plt.subplots(1, len(cs), figsize=(5.5 * len(cs), 4), squeeze=False)
        lim = max(1.0, np.abs(grp.l2).max())
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
            if 128 in hs:
                ax.axvline(hs.index(128) + 0.5, color="k", ls="--", lw=1)
        axes[0, 0].set_ylabel("scale")
        fig.colorbar(im, ax=axes[0], label="log2(gather / matmul)  (blue: matmul faster, red: gather faster)")
        fig.suptitle(f"{p}, {d}; dashed line: PR #4418 limit (128)")
        fig.savefig(fig_dir / f"heatmap_{p}_{d}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def compile_effect(g: pd.DataFrame, fig_dir: Path) -> None:
    g = g.dropna(subset=["gather", "gather_compiled"]).assign(sp=lambda d: d.gather / d.gather_compiled)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), sharey=True)
    for ax, p in zip(axes, ["fwd", "fwd_bwd"]):
        s = g[g["pass"] == p]
        ax.scatter(s.H, s.sp, s=8, alpha=0.4)
        m = s.groupby("H").sp.median()
        ax.plot(m.index, m.values, "r-o", ms=3, label="median")
        ax.axhline(1, color="k", lw=0.8)
        ax.set(xscale="log", title=p, xlabel="H = W")
    fig.suptitle("gather_compiled compiles the forward only; in fwd_bwd the backward runs uncompiled", fontsize=9)
    axes[0].set_ylabel("gather / gather_compiled")
    axes[0].legend()
    fig.savefig(fig_dir / "compile_effect.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def real_workloads(r: pd.DataFrame, fig_dir: Path) -> None:
    r = r.dropna(subset=["gather"])
    if r.empty:
        return
    labels = r.assign(label=lambda d: d.name + " B=" + d.B.astype(str) + " " + d.dtype.str.replace("float", "f"))
    labels = labels.sort_values(["name", "B"])
    names = list(dict.fromkeys(labels.label))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.3 * len(names) + 2), sharey=True)
    for ax, p in zip(axes, ["fwd", "fwd_bwd"]):
        s = labels[labels["pass"] == p].set_index("label").reindex(names)
        for k, v in enumerate(OTHERS):
            ax.barh(np.arange(len(names)) + (k - 1) * 0.27, s[v] / s.gather, height=0.27, label=v)
        ax.axvline(1, color="k", lw=0.8)
        ax.set(xscale="log", title=p, xlabel="time / gather time (below 1: faster)")
    axes[0].set_yticks(range(len(names)), names, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=7)
    fig.savefig(fig_dir / "real_workloads.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def gate_regret(w: pd.DataFrame) -> pd.DataFrame:
    """Cost of the PR #4418 gate, using the gather and matmul rows (same code the gate would run)."""
    d = w.dropna(subset=["gather", "matmul"]).copy()
    d["best"] = d[["gather", "matmul"]].min(axis=1)
    d["gate_matmul"] = (d.scale >= 1) & (d.H <= PR_MAX_INPUT) & (d.W <= PR_MAX_INPUT)
    d["chosen"] = np.where(d.gate_matmul, d.matmul, d.gather)
    d["regret_ms"] = d.chosen - d.best
    d["ratio"] = d.chosen / d.best
    d["gate_wrong"] = d.ratio > SLOW
    cols = [*WORKLOAD, "gather", "matmul", "gather_compiled", "best", "chosen", "regret_ms", "ratio", "gate_matmul", "gate_wrong"]
    return d[cols].sort_values("regret_ms", ascending=False)


def read_confirm(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # truncated last line from a crash
    return pd.DataFrame(rows)


def confirm_vs_sweep(df: pd.DataFrame, conf: pd.DataFrame) -> pd.DataFrame:
    """Per-variant absolute times, fresh-process median against the sweep."""
    key = ["name", "B", "dtype", "pass", "variant"]
    fresh = conf.groupby(key).ms_median.agg(fresh_ms="median", runs="size").reset_index()
    t = fresh.merge(df.dropna(subset=["ms_median"])[[*key, "ms_median"]].rename(columns={"ms_median": "sweep_ms"}), on=key)
    t["gap_pct"] = (t.fresh_ms / t.sweep_ms - 1) * 100
    return t.sort_values("gap_pct", key=abs, ascending=False)


def winner(r: pd.Series) -> pd.Series:
    """r = gather / matmul. matmul wins above 1/MARGIN, gather below MARGIN."""
    return pd.Series(np.where(r > 1 / MARGIN, "matmul", np.where(r < MARGIN, "gather", "tie")), index=r.index)


def winner_check(t: pd.DataFrame) -> pd.DataFrame:
    """Per (name, B, dtype, pass): gather / matmul ratio and winner, fresh against sweep."""
    p = t.pivot_table(index=["name", "B", "dtype", "pass"], columns="variant", values=["fresh_ms", "sweep_ms"])
    w = pd.DataFrame({
        "r_fresh": p["fresh_ms"]["gather"] / p["fresh_ms"]["matmul"],
        "r_sweep": p["sweep_ms"]["gather"] / p["sweep_ms"]["matmul"],
    }).dropna().reset_index()
    w["winner_fresh"], w["winner_sweep"] = winner(w.r_fresh), winner(w.r_sweep)
    w["agree"] = w.winner_fresh == w.winner_sweep
    w["opposite"] = w.winner_fresh.isin(["matmul", "gather"]) & w.winner_sweep.isin(["matmul", "gather"]) & ~w.agree
    w["ratio_gap_pct"] = np.abs(np.log(w.r_fresh / w.r_sweep)) * 100
    return w


def confirm_text(t: pd.DataFrame) -> str:
    if t.empty:
        return "(no confirm.jsonl)\n"
    w = winner_check(t)
    g = t[t.variant == "gather"]
    text = (
        "Fresh-process runs include a cold start, so absolute times are higher. "
        f"The median absolute gather time gap is {g.gap_pct.abs().median():.1f}%. "
        "The check therefore compares the ratio gather / matmul and the winner (same 0.9 margin).\n\n"
        f"- Configs compared: {len(w)}\n"
        f"- Same winner: {int(w.agree.sum())}\n"
        f"- Opposite winners (one says matmul, the other gather): {int(w.opposite.sum())}\n"
        f"- Ratio gap |log(r_fresh / r_sweep)|: median {w.ratio_gap_pct.median():.1f}%, "
        f"90th percentile {w.ratio_gap_pct.quantile(0.9):.1f}%\n\n"
        "Configs with a different winner:\n\n"
    )
    return text + md(w[~w.agree].drop(columns="agree"))


def candidate_rule(co: pd.DataFrame) -> str:
    if co.empty:
        return "(no grid data)\n"
    t = co.assign(
        wins=co.apply(lambda r: f"{r.status} (to H={int(r.max_H)})" if r.status != "partial"
                      else f"H {int(r.first_win_H)}..{int(r.last_win_H)} of {int(r.max_H)}" + ("" if r.contiguous else " (gaps)"), axis=1),
        col=co["pass"] + " " + co.dtype,
    ).pivot_table(index=["scale", "C"], columns="col", values="wins", aggfunc="first").fillna("-").reset_index()
    return md(t)


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
    for v in ["gather", *OTHERS]:
        if v not in wide:
            wide[v] = np.nan
    grid_w, real_w = wide[wide.set == "grid"], wide[wide.set == "real"]

    co = crossover(grid_w)
    co.to_csv(out / "crossover.csv", index=False)
    heatmaps(grid_w, fig_dir)
    compile_effect(grid_w, fig_dir)
    real_workloads(real_w, fig_dir)
    gr = gate_regret(wide)
    gr.to_csv(out / "gate_regret.csv", index=False)

    ce = (grid_w.gather / grid_w.gather_compiled).groupby(grid_w["pass"]).median()
    bad = df[df.valid != True].copy()  # noqa: E712
    err = bad["error"].fillna("") if "error" in bad else pd.Series("", index=bad.index)
    bad["error"] = np.where(err.str.contains("OutOfMemory|Insufficient Memory"), "OOM", np.where(err == "", "failed check", err.str[:60]))
    bad = bad.groupby(["name", "dtype", "pass", "variant", "error"]).size().reset_index(name="rows")
    conf = args.inp.parent / "confirm.jsonl"
    cg = confirm_vs_sweep(df, read_confirm(conf)) if conf.exists() else pd.DataFrame()
    if len(cg):
        cg.to_csv(out / "confirm_vs_sweep.csv", index=False)
    ge = df[df["pass"] == "fwd_bwd"].dropna(subset=["grad_rel_err"]) if "grad_rel_err" in df else df.iloc[:0]
    gstats = ge.groupby(["variant", "dtype"]).grad_rel_err.agg(median="median", max="max").reset_index()
    gtop = ge.sort_values("grad_rel_err", ascending=False).head(10)[["name", "B", "scale", "C", "dtype", "variant", "grad_rel_err"]]
    ge.drop(columns=["ms_repeats", "source"], errors="ignore").to_csv(out / "grad_error.csv", index=False)
    same = grid_w.dropna(subset=["gather", "matmul", "pr4418"])
    ref_col = np.where((same.scale >= 1) & (same.H <= PR_MAX_INPUT), same.matmul, same.gather)
    noise = np.abs(np.log2(same.pr4418 / ref_col))
    noise_txt = (f"pr4418 runs the same code as matmul (scale >= 1, H <= 128) or as gather (all other cases). "
                 f"Over {len(same)} grid configs, the median difference is {2 ** noise.median() - 1:.1%} and the "
                 f"90th percentile is {2 ** noise.quantile(0.9) - 1:.1%}, 95th percentile is {2 ** noise.quantile(0.95) - 1:.1%} "
                 f"and the largest is {2 ** noise.max() - 1:.1%}. The 10% margin is near the 90th percentile, "
                 "so a single win or tie close to the margin can be noise.\n"
                 if len(same) else "(no grid data)\n")
    top = gr.head(10)[["name", "B", "dtype", "pass", "gather", "matmul", "chosen", "regret_ms", "ratio", "gate_wrong"]]
    status = co.status.value_counts().to_dict() if len(co) else {}
    (out / "summary.md").write_text(
        f"# Summary\n\nSource: `{args.inp.name}`\n\n"
        f"- Rows: {len(df)} ({n_bad} invalid and dropped)\n"
        f"- Grid configs: {len(grid_w)}; real configs: {len(real_w)}\n"
        f"- Gate wrong (PR #4418 would pick a method more than 10% slower than the best of gather and matmul): {int(gr.gate_wrong.sum())} of {len(gr)}\n"
        "- All comparisons use eager (uncompiled) `gather`. A method wins only if it is below 0.9 x the other time; "
        "otherwise the size counts as a tie.\n"
        f"- Crossover groups (pass, scale, C, dtype): {status}\n"
        f"- Median gather / gather_compiled (grid): {ce.round(2).to_dict()}. "
        "`gather_compiled` compiles the forward only. In `fwd_bwd` the backward runs uncompiled, "
        "so this table does not show what compiling the backward would do.\n\n"
        f"## Invalid rows\n\n{md(bad)}\n"
        f"## Matmul wins over gather, by scale and C\n\n"
        "`always` and `never` cover all sizes up to the largest size that fits the memory cap (`max_H`). "
        "Otherwise the sizes where matmul wins are listed. "
        "`(gaps)` means gather clearly wins at a size between the first and last matmul win.\n\n"
        f"{candidate_rule(co)}\n"
        "## Gradient error against a float32 reference\n\n"
        "Relative error is the largest absolute difference divided by the largest absolute reference gradient. "
        "All rows with a gradient are included, also rows marked invalid.\n\n"
        f"{md(gstats)}\nTen worst rows:\n\n{md(gtop)}\n"
        f"## Noise control\n\n{noise_txt}\n"
        f"## Top 10 gate regrets (ms lost by the PR #4418 choice against the best of gather and matmul)\n\n{md(top)}\n"
        f"## Fresh-process check against the sweep (`confirm.jsonl`)\n\n"
        f"{confirm_text(cg)}\n"
        "## Candidate gate rule\n\n"
        "Read the rule from the table above: use matmul for a given scale and C only for H up to the "
        "largest winning size, and only where the table shows one block of wins. "
        "Where the table shows gaps or differs between dtypes, the data does not support one simple limit.\n"
    )
    print(f"wrote outputs to {out}")


if __name__ == "__main__":
    main()
