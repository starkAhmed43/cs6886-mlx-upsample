"""Regenerate the report figures at their printed size, with print-readable fonts.

Reads only committed experiment results (no GPU, no benchmarks) and writes the three
generated PNGs in `report/figures/` at their printed width in the report
(6.5in text width), with font sizes in points and dpi=300, so tick labels, axis
labels, titles and legends print at readable sizes when the figures are placed
at that width.

Run: python src/report/make_figures.py
"""

import importlib.util
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "report" / "figures"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_crossover_heatmap(tmp_dir: Path) -> Path:
    """01: heatmap_fwd_float32.png, at 0.8\\linewidth (~5.2in) and about 2.7:1."""
    analyze = _load("crossover_analyze", ROOT / "experiments/01_crossover/analyze.py")
    _, wide, _ = analyze.load_wide(ROOT / "experiments/01_crossover/results/sweep.jsonl")
    grid_w = wide[wide.set == "grid"]
    width = 5.2
    n_cs = 3  # C in {4, 32, 256} for pass=fwd, dtype=float32 -- see the groupby below
    analyze.heatmaps(
        grid_w, tmp_dir,
        col_width=width / n_cs, height=width / 2.7,
        dpi=300, fontsize=7,
        show_cell_numbers=False,  # too narrow per cell at this print size to stay >=5.5pt and non-overlapping
        cbar_label="log2(gather / matmul)",  # the full explanation is already in the report caption
        cbar_fraction=0.06, cbar_pad=0.08,
        bbox_inches=None,  # keep the saved size equal to figsize, so the fonts print at their set size
        constrained_layout=True,  # fits titles, ticks and colorbar within the exact figsize with no clipping
    )
    return tmp_dir / "heatmap_fwd_float32.png"


def make_rowsize(tmp_dir: Path) -> Path:
    """03: rowsize.png, at 0.55\\linewidth (~3.6in), 3.6x2.4in to match the roofline panel beside it."""
    rowsize = _load("rowsize_mod", ROOT / "experiments/03_tiling/rowsize.py")
    wide = rowsize.load_wide(ROOT / "experiments/03_tiling/results/rowsize.jsonl")
    width, height = 3.6, 2.4
    n_scales = len(wide.scale.unique())  # scale in {2, 4}
    out_path = tmp_dir / "rowsize.png"
    rowsize.plot_rowsize(
        wide, out_path,
        col_width=width / n_scales, height=height,
        dpi=300, fontsize=7,
        bbox_inches=None,  # keep the saved size equal to figsize, so the fonts print at their set size
        suptitle=None,  # the caption already says what the figure shows
        sharey=True,  # only the left panel gets a "GB/s" label and y ticks
        shared_xlabel=True, xlabel="input row size (bytes)", xlabel_y=0.03,
        shared_legend=True, legend_ncol=3, legend_fontsize=7,
        legend_loc="upper center", legend_bbox_to_anchor=(0.5, 1.0),
        top_margin=0.16,  # reserve room above the panel titles for the shared legend
        bottom_margin=0.16,  # reserve room below the panels for the x tick labels and shared x label
    )
    return out_path


def make_roofline_single(tmp_dir: Path) -> Path:
    """05: roofline_single.png, width fixed at 2.7in; legend sits below the axes in 2 columns."""
    roofline05 = _load("roofline05", ROOT / "experiments/05_final_benchmark/roofline.py")
    table = pd.read_csv(ROOT / "experiments/05_final_benchmark/results/table.csv")
    rf = roofline05.build_table(table)
    width, height = 2.7, 3.4  # axes keep ~2in height; the rest is the title and the 4-row, 2-col legend below
    out_path = tmp_dir / "roofline_single.png"
    roofline05.single_panel_figure(
        rf, out_path, roofline05.VARIANT_COLORS, roofline05.PASS_MARKERS,
        legend_fontsize=7, figsize=(width, height), dpi=300, fontsize=8,
        legend_outside=True, legend_ncol=2, legend_bottom=0.147,
        legend_kwargs={
            "handlelength": 0.6, "handletextpad": 0.1, "labelspacing": 0.15, "columnspacing": 0.1, "borderpad": 0.1,
        },
    )
    return out_path


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = FIG_DIR / "_tmp"
    tmp_dir.mkdir(exist_ok=True)
    try:
        sources = {
            "crossover_heatmap.png": make_crossover_heatmap(tmp_dir),
            "rowsize.png": make_rowsize(tmp_dir),
            "roofline_single.png": make_roofline_single(tmp_dir),
        }
        for name, src in sources.items():
            shutil.copyfile(src, FIG_DIR / name)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"wrote {', '.join(sources)} to {FIG_DIR}")


if __name__ == "__main__":
    main()
