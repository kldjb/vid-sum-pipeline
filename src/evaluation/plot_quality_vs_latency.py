import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path

# One colour per configuration so both panels stay visually consistent.
CONFIG_COLORS = {
    "visual_baseline": "#4C72B0",
    "dialogue_baseline": "#DD8452",
    "semantic_baseline": "#55A868",
    "ft_bimodal_vector": "#C44E52",
    "fs_bimodal_vector": "#8172B2",
    "ts_bimodal_vector": "#937860",
    "trimodal_vector": "#DA8BC3",
    "full_graphrag": "#8C8C8C",
    "visual_baseline_graph": "#CCB974",
    "ft_bimodal_vector_graph": "#64B5CD",
}

# Colour used for any configuration not listed in CONFIG_COLORS above.
DEFAULT_CONFIG_COLOR = "#E24A33"

QUERY_SOURCE_MARKERS = {
    "self": "o",
    "cross_annotator": "^",
}


def get_config_color(config: str) -> str:
    """
    Look up a configuration's plot colour.
    """
    if config not in CONFIG_COLORS:
        print(
            f"WARNING: '{config}' has no entry in CONFIG_COLORS - plotting "
            "with the fallback colour. Add it to CONFIG_COLORS for a "
            "stable, distinguishable colour of its own."
        )
    return CONFIG_COLORS.get(config, DEFAULT_CONFIG_COLOR)

def load_summary(csv_path: Path) -> pd.DataFrame:
    """
    Load the raw per-video results and aggregate to one row per
    (configuration, query_source) - mean latency and mean quality metrics.
    """
    df = pd.read_csv(csv_path)

    required = {"configuration", "latency_sec"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{csv_path} is missing expected column(s): {missing}. "
            "Is this the right CSV (videoxum_ablation_results.csv)?"
        )
    group_cols = ["configuration"]
    if "query_source" in df.columns:
        group_cols.append("query_source")
    else:
        df["query_source"] = "self"
        group_cols.append("query_source")

    metric_cols = [c for c in ("f1_dilated", "vt_clipscore") if c in df.columns]
    if not metric_cols:
        raise ValueError(
            f"{csv_path} has neither 'f1_dilated' nor 'vt_clipscore' - "
            "nothing to plot on the quality axis."
        )

    summary = (
        df.groupby(group_cols)[["latency_sec"] + metric_cols]
        .mean()
        .reset_index()
    )
    return summary, metric_cols

def plot_panel(ax, summary: pd.DataFrame, metric: str, metric_label: str) -> None:
    for _, row in summary.iterrows():
        config = row["configuration"]
        source = row["query_source"]
        color = get_config_color(config)
        marker = QUERY_SOURCE_MARKERS.get(source, "o")

        ax.scatter(
            row["latency_sec"],
            row[metric],
            color=color,
            marker=marker,
            s=110,
            edgecolors="white",
            linewidths=0.8,
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Mean retrieval latency (s, log scale)")
    ax.set_ylabel(metric_label)
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.5)


def build_legend(fig, summary: pd.DataFrame) -> None:
    """
    Legend uses colour for configurations and shapes to distinguish query source.
    """
    present_configs = summary["configuration"].unique()
    # CONFIG_COLORS' own order first, then any config missing from it
    ordered_configs = [c for c in CONFIG_COLORS if c in present_configs]
    ordered_configs += [c for c in present_configs if c not in CONFIG_COLORS]

    config_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=get_config_color(config),
               markersize=8, label=config)
        for config in ordered_configs
    ]

    source_handles = []
    if summary["query_source"].nunique() > 1:
        source_handles = [
            Line2D([0], [0], marker=marker, color="#333333", linestyle="",
                   markersize=8, label=source)
            for source, marker in QUERY_SOURCE_MARKERS.items()
            if source in summary["query_source"].unique()
        ]

    fig.legend(
        handles=config_handles + source_handles,
        loc="lower center",
        ncol=min(4, len(config_handles) + len(source_handles)),
        bbox_to_anchor=(0.5, -0.05),
        fontsize=8,
        frameon=False,
    )

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default="videoxum_ablation_results.csv",
        help="Path to the ablation results CSV (default: videoxum_ablation_results.csv)",
    )
    parser.add_argument(
        "--output", default="quality_vs_latency.png",
        help="Output image path (default: quality_vs_latency.png)",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run evaluate_video.py's "
            "evaluate_extracted_dataset first to produce it."
        )

    summary, metric_cols = load_summary(csv_path)

    metric_labels = {
        "f1_dilated": "Mean dilated F1 (temporal retrieval quality)",
        "vt_clipscore": "Mean VT-CLIPScore (cross-modal alignment, 0-100)",
    }

    fig, axes = plt.subplots(1, len(metric_cols), figsize=(7 * len(metric_cols), 5.5))
    if len(metric_cols) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metric_cols):
        plot_panel(ax, summary, metric, metric_labels.get(metric, metric))

    calibration = "Calibrated" if "calibrated" in csv_path.stem else "Uncalibrated"
    fig.suptitle(f"{calibration} retrieval quality vs. latency across ablation configurations", fontsize=13)
    build_legend(fig, summary)
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    fig.savefig(f"{calibration}_{args.output}", dpi=200, bbox_inches="tight")
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
