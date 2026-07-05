from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
TABLES = ROOT.parent / "current_tables"
FIGURES = ROOT.parent / "current_figures"

DATASETS = ["synthetic", "uci_har", "air_quality"]
DATASET_LABELS = {
    "synthetic": "Synthetic",
    "uci_har": "UCI HAR",
    "air_quality": "Air Quality",
}

METHOD_LABELS = {
    "statistical_hier": "Statistical",
    "spectral_hier": "Spectral",
    "stat_spectral_hier": "Stat.+spectral",
    "tuned_tcn_ae_hier": "Tuned TCN-AE",
    "tuned_usad_ae_hier": "Tuned USAD-like AE",
    "gru_ae_hier": "GRU-AE",
    "hra_plus_spectral_hier": "HRA+spectral",
    "diagnostic_signature_only_hier": "HRA-Core signature",
    "residual_summaries_only_hier": "Residual summaries",
    "attribution_scores_only_hier": "Attribution scores",
    "indices_only_hier": "Indices only",
    "statistical_flat": "Statistical flat",
    "spectral_flat": "Spectral flat",
    "hra_plus_spectral_flat": "HRA+spectral flat",
}

HIERARCHY_LABELS = {
    "statistical_flat": "Statistical flat",
    "statistical_hier": "Statistical hier.",
    "spectral_flat": "Spectral flat",
    "spectral_hier": "Spectral hier.",
    "hra_plus_spectral_flat": "HRA+spectral flat",
    "hra_plus_spectral_hier": "HRA+spectral hier.",
}

PALETTE = [
    "#5B7FA3",
    "#6EA6A1",
    "#86A65A",
    "#D19A3A",
    "#A983A3",
    "#8D7468",
    "#7E83BF",
]
HATCHES = ["", "///", "\\\\\\", "xx", "--", "++", ".."]


def setup_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save(fig, name, tight=True):
    for ext in ("pdf", "png"):
        kwargs = {"dpi": 300}
        if tight:
            kwargs["bbox_inches"] = "tight"
        fig.savefig(FIGURES / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def values_for(df, systems, metric, err_metric=None):
    vals = []
    errs = []
    for dataset in DATASETS:
        ds_vals = []
        ds_errs = []
        for system in systems:
            row = df[(df["dataset"] == dataset) & (df["system"] == system)]
            ds_vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
            ds_errs.append(float(row[err_metric].iloc[0]) if err_metric and len(row) else 0.0)
        vals.append(ds_vals)
        errs.append(ds_errs)
    return np.asarray(vals), np.asarray(errs)


def grouped_bar(df, systems, metric, err_metric, ylabel, output_name, legend_cols=4, labels=None):
    labels = labels or METHOD_LABELS
    vals, errs = values_for(df, systems, metric, err_metric)
    fig, ax = plt.subplots(figsize=(10.6, 4.7))
    x = np.arange(len(DATASETS))
    width = 0.82 / len(systems)
    for i, system in enumerate(systems):
        offset = (i - (len(systems) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            vals[:, i],
            width,
            yerr=errs[:, i],
            capsize=2,
            color=PALETTE[i % len(PALETTE)],
            edgecolor="#303030",
            linewidth=0.45,
            hatch=HATCHES[i % len(HATCHES)],
            label=labels.get(system, system),
        )
        for patch in bars:
            patch.set_linewidth(0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([DATASET_LABELS[d] for d in DATASETS])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", color="#D8D8D8", linewidth=0.55)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=legend_cols,
        frameon=False,
        columnspacing=1.3,
        handletextpad=0.5,
    )
    fig.tight_layout()
    save(fig, output_name)


def plot_baseline():
    systems = [
        "statistical_hier",
        "spectral_hier",
        "stat_spectral_hier",
        "tuned_tcn_ae_hier",
        "tuned_usad_ae_hier",
        "gru_ae_hier",
        "hra_plus_spectral_hier",
    ]
    df = pd.read_csv(TABLES / "baseline_strengthened_summary_current.csv")
    grouped_bar(
        df,
        systems,
        "mean_macro_f1",
        "ci95_macro_f1",
        "Source-level macro-F1",
        "baseline_strengthened_macro_f1",
        legend_cols=4,
    )


def plot_ablation():
    systems = [
        "diagnostic_signature_only_hier",
        "residual_summaries_only_hier",
        "attribution_scores_only_hier",
        "indices_only_hier",
        "hra_plus_spectral_hier",
    ]
    df = pd.read_csv(TABLES / "ablation_feature_blocks_current.csv")
    grouped_bar(
        df,
        systems,
        "mean_macro_f1",
        "ci95_macro_f1",
        "Source-level macro-F1",
        "feature_block_ablation",
        legend_cols=5,
    )


def plot_hierarchy():
    systems = [
        "statistical_flat",
        "statistical_hier",
        "spectral_flat",
        "spectral_hier",
        "hra_plus_spectral_flat",
        "hra_plus_spectral_hier",
    ]
    df = pd.read_csv(TABLES / "hierarchy_vs_flat_current.csv")
    grouped_bar(
        df,
        systems,
        "mean_macro_f1",
        "ci95_macro_f1",
        "Source-level macro-F1",
        "hierarchy_vs_flat",
        legend_cols=3,
        labels=HIERARCHY_LABELS,
    )


def plot_fault_type():
    df = pd.read_csv(TABLES / "fault_type_localization_current.csv")
    local_types = [
        "offset_drift",
        "scale_drift",
        "trend_drift",
        "noise_floor",
        "saturation",
        "dropout",
        "time_lag",
    ]
    pretty = {
        "offset_drift": "Offset drift",
        "scale_drift": "Scale drift",
        "trend_drift": "Trend drift",
        "noise_floor": "Noise-floor",
        "saturation": "Saturation",
        "dropout": "Dropout",
        "time_lag": "Time lag",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.9), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        sub = df[df["dataset"] == dataset]
        vals = []
        for fault in local_types:
            row = sub[sub["fault_type"] == fault]
            vals.append(float(row["localization_top1"].iloc[0]) if len(row) else np.nan)
        y = np.arange(len(local_types))
        bars = ax.barh(
            y,
            vals,
            color=[PALETTE[i % len(PALETTE)] for i in range(len(local_types))],
            edgecolor="#303030",
            linewidth=0.45,
        )
        for i, bar in enumerate(bars):
            bar.set_hatch(HATCHES[i % len(HATCHES)])
        for yi, val in zip(y, vals):
            if np.isfinite(val):
                x_text = min(val + 0.025, 0.94)
                ha = "left" if val < 0.90 else "right"
                ax.text(x_text, yi, f"{val:.2f}", va="center", ha=ha, fontsize=8.2)
        ax.set_xlim(0, 1.0)
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.grid(axis="x", color="#D8D8D8", linewidth=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels([pretty[f] for f in local_types])
        ax.tick_params(axis="y", labelleft=True, pad=3)
    axes[0].invert_yaxis()
    fig.supxlabel("Top-1 affected-channel localization", y=0.02)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.18, top=0.88, wspace=0.36)
    save(fig, "fault_type_breakdown", tight=False)


def main():
    FIGURES.mkdir(exist_ok=True)
    setup_style()
    plot_baseline()
    plot_ablation()
    plot_hierarchy()
    plot_fault_type()


if __name__ == "__main__":
    main()
