"""Shared roofline model for the bilinear-upsample experiments.

Built only from existing timing results (`ms_median` columns already in each experiment's
output) -- this module runs no GPU timing of its own.

Hardware model (Apple M1, 8-core GPU; the only inputs not derived from results):

- DRAM bandwidth peak: 68.25 GB/s.
- fp32 compute peak: 2.6 TFLOP/s.
- Ridge point (compute/bandwidth): 2.6e12 / 68.25e9 ~= 38.1 FLOP/byte.

Traffic model (minimum bytes a correct implementation must move; used for every variant in
every experiment so the plots show wasted traffic as lower achieved GB/s, not a different
roofline):

- forward: read input once, write output once. For the input read, a correct bilinear kernel
  only touches at most 2 input indices per output index on each axis, so for downscaling
  (oh < H or ow < W) the forward input read is capped at `min(H, 2*oh) * min(W, 2*ow)` pixels
  per (batch, channel), not the full `H*W`. For upscaling this cap is not binding (every input
  pixel is read at least once), so it has no effect.
- forward+backward: forward bytes, plus read grad-output once (same size as the output) and
  write grad-input once. The grad-input write always covers the full input shape (every input
  position gets a gradient, even where it is zero), so it is not subject to the downscaling
  read cap above.
- Element size is 2 bytes for float16, 4 for float32. Input/grad-input bytes always use the
  input dtype's size. Output/grad-output element size is the variant's actual output dtype
  (callers pass this in as `out_elem`, since it depends on the variant and sometimes the
  experiment -- see each roofline.py for the rule it uses).
- Cache-line granularity is ignored; bytes are counted at the element level only.

FLOPs models:

- Elementwise bilinear (gather-style kernels: `gather`, `gather_compiled`, `fused*`, and
  `pr4418` when it takes its gather path): each output element is 3 lerps, 3 flops each
  (sub, mul, add) = 9 flops/output-element for the forward pass. The backward pass is treated
  as the same cost per output element, so `fwd_bwd` rows use 9 flops/output-element for each of
  the two passes (18 total).
- Matmul (`matmul`, and `pr4418` when it takes its matmul path -- see `uses_matmul`): the real
  dense FLOPs of `_interpolate_matmul` (`mlx_upsample.baselines.pr4418`), which runs one dense
  matmul per spatial axis, in order (height axis, then width axis). For axis i with input size
  N_i and output size M_i, contracted against the other dims at that point in the sequence, one
  matmul costs `2 * M_i * N_i * K_i` FLOPs (K_i = product of the other dims, which shrinks as
  earlier axes are resampled). The backward pass is approximated as 2x the forward cost (a dense
  matmul's backward is itself two matmuls of about the same size); this is documented, not
  derived from a measurement.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mlx_upsample.baselines.pr4418 import _MATMUL_MAX_INPUT_SIZE_LINEAR, _use_matmul
from mlx_upsample.workloads import Workload, out_hw

PEAK_BW_GBPS = 68.25  # GB/s
PEAK_COMPUTE_GFLOPS = 2.6e3  # GFLOP/s (2.6 TFLOP/s)
FLOPS_PER_OUTPUT_ELEM = 9


def dtype_bytes(dtype: str) -> int:
    return 2 if dtype == "float16" else 4


def _workload(row: pd.Series) -> Workload:
    return Workload("", "", "real", int(row["B"]), int(row["H"]), int(row["W"]), int(row["C"]), float(row["scale"]), bool(row["align_corners"]), row["dtype"])


def uses_matmul(row: pd.Series) -> bool:
    """Whether PR #4418's gate (`_use_matmul`) picks the matmul path for this row's shape."""
    return _use_matmul([int(row["H"]), int(row["W"])], (float(row["scale"]), float(row["scale"])), _MATMUL_MAX_INPUT_SIZE_LINEAR)


def min_bytes(row: pd.Series, out_elem: int) -> int:
    """Minimum bytes moved for one (workload, pass) row, given its output element size."""
    w = _workload(row)
    oh, ow = out_hw(w)
    in_elem = dtype_bytes(w.dtype)
    n_in_full = w.B * w.H * w.W * w.C
    n_in_read = w.B * min(w.H, 2 * oh) * min(w.W, 2 * ow) * w.C
    n_out = w.B * oh * ow * w.C

    fwd_bytes = n_in_read * in_elem + n_out * out_elem
    if row["pass"] == "fwd":
        return fwd_bytes
    # fwd_bwd: forward bytes, plus grad-output read (out_bytes) and grad-input write (full input size)
    return fwd_bytes + n_out * out_elem + n_in_full * in_elem


def elementwise_flops(row: pd.Series) -> int:
    """9 flops/output-element per pass (gather-style bilinear kernels)."""
    w = _workload(row)
    oh, ow = out_hw(w)
    n_out = w.B * oh * ow * w.C
    fwd_flops = FLOPS_PER_OUTPUT_ELEM * n_out
    return fwd_flops if row["pass"] == "fwd" else 2 * fwd_flops


def matmul_flops(row: pd.Series) -> int:
    """Dense FLOPs of `_interpolate_matmul`, in its axis order (height, then width)."""
    w = _workload(row)
    oh, ow = out_hw(w)
    fwd_flops = 2 * oh * w.H * (w.B * w.W * w.C) + 2 * ow * w.W * (w.B * oh * w.C)
    return fwd_flops if row["pass"] == "fwd" else 2 * fwd_flops


def build_table(table: pd.DataFrame, bytes_flops_fn) -> pd.DataFrame:
    """Roofline rows from a results table. `bytes_flops_fn(row) -> (min_bytes, flops)`."""
    rows = []
    for _, r in table.iterrows():
        min_b, flops = bytes_flops_fn(r)
        t_s = r["ms_median"] / 1e3
        gflops = flops / t_s / 1e9
        gbps = min_b / t_s / 1e9
        rows.append(
            {
                "name": r["name"],
                "B": r["B"],
                "dtype": r["dtype"],
                "pass": r["pass"],
                "variant": r["variant"],
                "flops": flops,
                "min_bytes": min_b,
                "ai": flops / min_b,
                "gflops": gflops,
                "gbps": gbps,
                "frac_of_peak_bw": gbps / PEAK_BW_GBPS,
            }
        )
    return pd.DataFrame(rows)


def _plot_roofline_panel(ax, rf: pd.DataFrame, variant_colors: dict, pass_markers: dict | None = None) -> None:
    """Draw the roofline scatter (achieved GFLOP/s vs arithmetic intensity) onto `ax`."""
    ai_lo, ai_hi = rf.ai.min() / 2, rf.ai.max() * 2
    ai_line = np.geomspace(ai_lo, ai_hi, 200)
    mem_roof = PEAK_BW_GBPS * ai_line
    roof = np.minimum(mem_roof, PEAK_COMPUTE_GFLOPS)
    ax.plot(ai_line, roof, "k-", lw=1, label=f"roofline ({PEAK_BW_GBPS:g} GB/s, {PEAK_COMPUTE_GFLOPS/1e3:g} TFLOP/s)")
    markers = pass_markers or {p: "o" for p in sorted(rf["pass"].unique())}
    for (variant, pas), g in rf.groupby(["variant", "pass"]):
        ax.scatter(g.ai, g.gflops, s=10, c=variant_colors[variant], marker=markers.get(pas, "o"), alpha=0.7, label=f"{variant} {pas}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("achieved GFLOP/s")
    ax.set_title("Roofline")


def single_panel_figure(
    rf: pd.DataFrame,
    path: Path,
    variant_colors: dict,
    pass_markers: dict | None = None,
    legend_fontsize: float = 5.0,
    figsize: tuple = (3.4, 2.9),
    dpi: int = 150,
    fontsize: float = 8,
    legend_kwargs: dict | None = None,
    legend_outside: bool = False,
    legend_ncol: int = 1,
    legend_bottom: float | None = None,
) -> None:
    """Roofline-only figure at about half text width, for use in a report.

    By default the legend sits inside the axes, in the upper-left area where the roofline
    scatter is typically sparse (low arithmetic intensity means low achieved GFLOP/s). With
    `legend_outside=True` it is drawn instead as one row below the axes (`legend_bottom`
    reserves that fraction of the figure height), so it never covers data.
    """
    plt.rcParams.update({"font.size": fontsize})
    fig, ax = plt.subplots(figsize=figsize)
    _plot_roofline_panel(ax, rf, variant_colors, pass_markers)
    if legend_outside:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=legend_fontsize, loc="lower center", ncol=legend_ncol,
                   bbox_to_anchor=(0.5, 0.0), **(legend_kwargs or {}))
        fig.tight_layout(rect=(0, legend_bottom or 0, 1, 1))
    else:
        ax.legend(fontsize=legend_fontsize, loc="upper left", **(legend_kwargs or {}))
        fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def figure(
    rf: pd.DataFrame,
    path: Path,
    variant_colors: dict,
    order: list,
    pass_markers: dict | None = None,
    legend_ncol: int = 4,
    legend_fontsize: float = 6.5,
) -> None:
    """Two-panel figure: roofline (left) and achieved-bandwidth-fraction by variant (right).

    The panel-1 legend (roofline line + one entry per variant/pass) is drawn outside the axes,
    as one shared legend below both panels, so it never covers data points.
    """
    plt.rcParams.update({"font.size": 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel 1: roofline
    _plot_roofline_panel(ax1, rf, variant_colors, pass_markers)

    # Panel 2: achieved bandwidth as a fraction of peak, by variant
    data = [rf[rf.variant == v].frac_of_peak_bw.values for v in order]
    ax2.boxplot(data, tick_labels=order, showfliers=True, widths=0.5)
    for i, v in enumerate(order):
        x = np.random.default_rng(0).normal(i + 1, 0.05, size=len(data[i]))
        ax2.scatter(x, data[i], s=4, c=variant_colors[v], alpha=0.5)
    ax2.set_ylabel(f"achieved BW / peak ({PEAK_BW_GBPS:g} GB/s)")
    ax2.set_title("Bandwidth utilization")
    ax2.tick_params(axis="x", labelsize=7)

    # Shared legend below both panels, outside the axes, so it never covers data points.
    handles, labels = ax1.get_legend_handles_labels()
    ncol = min(legend_ncol, len(labels))
    nrows = -(-len(labels) // ncol)
    bottom = 0.02 + 0.055 * nrows
    fig.legend(handles, labels, fontsize=legend_fontsize, ncol=ncol, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, bottom, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)
