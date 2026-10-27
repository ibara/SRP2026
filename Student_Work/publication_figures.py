#!/usr/bin/env python3
"""Generate the four compact figures selected for the eight-page paper.

The script reads only finalized analysis CSVs. It does not recompute kernels,
classifiers, confidence intervals, or statistical tests.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent / "hardware_results/results/analysis"
CONFIG_ORDER = (
    "clamp_native10",
    "clamp_pca4",
    "fabiana_native8",
    "fabiana_pca4",
    "cic_pca4",
    "ember2018_pca4",
)
CONFIG_LABELS = {
    "clamp_native10": "CLaMP native-10",
    "clamp_pca4": "CLaMP PCA-4",
    "fabiana_native8": "MBR native-8",
    "fabiana_pca4": "MBR PCA-4",
    "cic_pca4": "CIC PCA-4",
    "ember2018_pca4": "EMBER2018 PCA-4",
}
MODE_ORDER = ("exact", "ideal_aer", "noisy_aer", "hardware")
MODE_LABELS = {
    "exact": "Exact",
    "ideal_aer": "Ideal Aer",
    "noisy_aer": "Noisy Aer",
    "hardware": "Hardware",
}
MODE_COLORS = {
    "exact": "#333333",
    "ideal_aer": "#0072B2",
    "noisy_aer": "#E69F00",
    "hardware": "#D55E00",
}
MODE_MARKERS = {
    "exact": "o",
    "ideal_aer": "s",
    "noisy_aer": "^",
    "hardware": "D",
}
CONFIG_COLORS = {
    configuration: plt.get_cmap("tab10")(index)
    for index, configuration in enumerate(CONFIG_ORDER)
}
CONFIG_MARKERS = dict(zip(CONFIG_ORDER, ("o", "s", "^", "D", "P", "X")))


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": [
                "Nimbus Roman",
                "Times New Roman",
                "Times",
                "DejaVu Serif",
            ],
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 6.7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def finish_figure(fig: plt.Figure, stem: str) -> None:
    for suffix in ("pdf", "png", "svg"):
        options = {"dpi": 400} if suffix == "png" else {}
        fig.savefig(
            ANALYSIS_DIR / f"{stem}.{suffix}",
            bbox_inches="tight",
            pad_inches=0.025,
            **options,
        )
    plt.close(fig)


def style_axis(axis: plt.Axes, grid_axis: str = "x") -> None:
    axis.grid(axis=grid_axis, color="#d5d5d5", linewidth=0.55, alpha=0.8)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)


def execution_performance_figure(intervals: pd.DataFrame) -> None:
    selected = intervals[
        (intervals["metric"] == "balanced_accuracy")
        & intervals["configuration"].isin(CONFIG_ORDER)
    ]
    fig, axis = plt.subplots(figsize=(7.15, 3.05))
    base_positions = np.arange(len(CONFIG_ORDER), dtype=float)
    offsets = dict(zip(MODE_ORDER, (-0.27, -0.09, 0.09, 0.27)))
    for mode in MODE_ORDER:
        rows = (
            selected[selected["mode"] == mode]
            .set_index("configuration")
            .loc[list(CONFIG_ORDER)]
        )
        estimates = rows["estimate"].to_numpy(float)
        low = rows["ci_low"].to_numpy(float)
        high = rows["ci_high"].to_numpy(float)
        axis.errorbar(
            estimates,
            base_positions + offsets[mode],
            xerr=np.vstack((estimates - low, high - estimates)),
            fmt=MODE_MARKERS[mode],
            color=MODE_COLORS[mode],
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            elinewidth=0.9,
            capsize=1.8,
            label=MODE_LABELS[mode],
        )
    axis.set_yticks(base_positions, [CONFIG_LABELS[item] for item in CONFIG_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0.52, 1.005)
    axis.set_xlabel("Balanced accuracy")
    style_axis(axis, "x")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.tight_layout()
    finish_figure(fig, "paper_figure1_execution_performance")


def kernel_distortion_figure(aggregate: pd.DataFrame) -> None:
    fig, axis = plt.subplots(figsize=(7.15, 2.85))
    x = np.arange(len(MODE_ORDER), dtype=float)
    for configuration in CONFIG_ORDER:
        rows = (
            aggregate[aggregate["configuration"] == configuration]
            .set_index("mode")
            .loc[list(MODE_ORDER)]
        )
        mean = rows["test_relative_frobenius_error_mean"].to_numpy(float)
        sd = rows[
            "test_relative_frobenius_error_standard_deviation"
        ].to_numpy(float)
        axis.errorbar(
            x,
            mean,
            yerr=sd,
            marker=CONFIG_MARKERS[configuration],
            color=CONFIG_COLORS[configuration],
            linewidth=1.15,
            elinewidth=0.75,
            capsize=1.8,
            markersize=4.2,
            markeredgecolor="white",
            markeredgewidth=0.3,
            label=CONFIG_LABELS[configuration],
        )
    axis.set_xticks(x, [MODE_LABELS[item] for item in MODE_ORDER])
    axis.set_xlim(-0.12, len(MODE_ORDER) - 0.88)
    axis.set_ylim(-0.02, 0.76)
    axis.set_ylabel("Relative Frobenius error")
    axis.set_xlabel("Execution environment")
    style_axis(axis, "y")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        handletextpad=0.35,
        columnspacing=0.9,
    )
    fig.tight_layout()
    finish_figure(fig, "paper_figure2_kernel_distortion")


def quantum_classical_figure(
    aggregate: pd.DataFrame, classical: pd.DataFrame
) -> None:
    quantum = aggregate[aggregate["mode"].isin(("exact", "hardware"))]
    method_specs = (
        ("linear", "Linear SVC", "#999999", "v"),
        ("rbf", "RBF SVC", "#009E73", "P"),
        ("exact", "Exact quantum", MODE_COLORS["exact"], "o"),
        ("hardware", "Hardware quantum", MODE_COLORS["hardware"], "D"),
    )
    method_keys = tuple(item[0] for item in method_specs)
    offsets = dict(zip(method_keys, (-0.27, -0.09, 0.09, 0.27)))
    base_positions = np.arange(len(CONFIG_ORDER), dtype=float)
    fig, axis = plt.subplots(figsize=(7.15, 3.05))
    for key, label, color, marker in method_specs:
        if key in {"linear", "rbf"}:
            rows = classical[classical["kernel"] == key]
        else:
            rows = quantum[quantum["mode"] == key]
        rows = rows.set_index("configuration").loc[list(CONFIG_ORDER)]
        mean = rows["balanced_accuracy_mean"].to_numpy(float)
        sd = rows["balanced_accuracy_standard_deviation"].to_numpy(float)
        axis.errorbar(
            mean,
            base_positions + offsets[key],
            xerr=sd,
            fmt=marker,
            color=color,
            markersize=4.0,
            markeredgecolor="white",
            markeredgewidth=0.35,
            elinewidth=0.85,
            capsize=1.7,
            label=label,
        )
    axis.set_yticks(base_positions, [CONFIG_LABELS[item] for item in CONFIG_ORDER])
    axis.invert_yaxis()
    axis.set_xlim(0.48, 1.005)
    axis.set_xlabel(r"Balanced accuracy, mean $\pm$ split SD")
    style_axis(axis, "x")
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        handletextpad=0.35,
        columnspacing=1.0,
    )
    fig.tight_layout()
    finish_figure(fig, "paper_figure3_quantum_classical")


def circuit_cost_figure(run_metrics: pd.DataFrame, circuits: pd.DataFrame) -> None:
    primary = run_metrics[run_metrics["repeat"] == 0]
    hardware = primary[primary["mode"] == "hardware"].copy()
    exact = primary[primary["mode"] == "exact"][
        ["configuration", "fold", "balanced_accuracy"]
    ].rename(columns={"balanced_accuracy": "exact_balanced_accuracy"})
    cost = hardware.merge(exact, on=["configuration", "fold"], validate="one_to_one")
    cz_by_width = circuits.set_index("logical_qubits")[
        "cz_per_fidelity_circuit"
    ].to_dict()
    cost["cz_count"] = cost["qubits"].map(cz_by_width)
    cost["performance_change"] = (
        cost["balanced_accuracy"] - cost["exact_balanced_accuracy"]
    )
    r_performance = cost["cz_count"].corr(cost["performance_change"])
    r_distortion = cost["cz_count"].corr(
        cost["test_relative_frobenius_error"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.75))
    for configuration in CONFIG_ORDER:
        rows = cost[cost["configuration"] == configuration]
        options = {
            "s": 27,
            "marker": CONFIG_MARKERS[configuration],
            "color": CONFIG_COLORS[configuration],
            "edgecolors": "white",
            "linewidths": 0.35,
            "zorder": 3,
        }
        axes[0].scatter(
            rows["cz_count"],
            rows["performance_change"],
            label=CONFIG_LABELS[configuration],
            **options,
        )
        axes[1].scatter(
            rows["cz_count"],
            rows["test_relative_frobenius_error"],
            **options,
        )
    axes[0].axhline(0, color="#333333", linewidth=0.75)
    axes[0].set_ylabel(r"Hardware $-$ exact balanced accuracy")
    axes[1].set_ylabel("Hardware kernel relative error")
    axes[0].text(
        0.03,
        0.95,
        rf"Pearson $r={r_performance:.3f}$",
        transform=axes[0].transAxes,
        va="top",
    )
    axes[1].text(
        0.03,
        0.95,
        rf"Pearson $r={r_distortion:.3f}$",
        transform=axes[1].transAxes,
        va="top",
    )
    for axis in axes:
        axis.set_xlabel("CZ gates per fidelity circuit")
        axis.set_xticks((22, 54, 70))
        style_axis(axis, "both")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    finish_figure(fig, "paper_figure4_circuit_cost")


def main() -> None:
    configure_matplotlib()
    aggregate = pd.read_csv(ANALYSIS_DIR / "aggregate_run_metrics.csv")
    intervals = pd.read_csv(ANALYSIS_DIR / "bootstrap_intervals.csv")
    classical = pd.read_csv(ANALYSIS_DIR / "aggregate_classical_baselines.csv")
    run_metrics = pd.read_csv(ANALYSIS_DIR / "run_metrics.csv")
    circuits = pd.read_csv(ANALYSIS_DIR / "circuit_resources.csv")
    execution_performance_figure(intervals)
    kernel_distortion_figure(aggregate)
    quantum_classical_figure(aggregate, classical)
    circuit_cost_figure(run_metrics, circuits)
    print(f"Wrote four paper figures to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()
