"""02: Generate static overview, PCA, and differential-expression figures."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.my_funcs import (
    CONDITION_MAP,
    DATA_DIR,
    FIGURE_DIR,
    ANALYTE_COLUMNS,
    complete_analyte_columns,
    pca_analytes,
    plot_condition,
    plot_cytokine,
    safe_filename,
    save_figure,
)


OUTPUT_DIR = FIGURE_DIR / "02_initial_viz"


def plot_pca_outputs(df: pd.DataFrame, analytes: list[str]) -> None:
    """Save PCA score and loading plots."""
    # including +/- IL1B
    _, fig = pca_analytes(df, analytes)
    save_figure(fig, OUTPUT_DIR / "pca" / "pca_all_samples.png")

    # only IL1B+ condition
    pca, fig = pca_analytes(df, analytes, il1b_pos_only=True)
    save_figure(fig, OUTPUT_DIR / "pca" / "pca_il1b_positive.png")

    loadings = pd.DataFrame(
        pca.components_.T,
        index=analytes,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
    )
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    for ax, component in zip(axs, ["PC1", "PC2"]):
        ordered = loadings[component].sort_values(key=abs, ascending=False)
        ax.bar(ordered.index, ordered.values)
        ax.set_title(f"{component} Loadings")
        ax.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "pca" / "pca_loadings_bars.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(loadings["PC1"], loadings["PC2"])
    for analyte, row in loadings.iterrows():
        ax.annotate(
            analyte, (row["PC1"], row["PC2"]), xytext=(4, 3),
            textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel("PC1 loading")
    ax.set_ylabel("PC2 loading")
    ax.set_title("Luminex PCA Loadings")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "pca" / "pca_loadings_scatter.png")


def plot_analyte_boxplots(df: pd.DataFrame, analytes: list[str]) -> None:
    """Replace the notebook widget with one saved figure per analyte."""
    df_pos = df[df["saw_il1b"]]
    df_neg = df[~df["saw_il1b"]]
    for cytokine in analytes:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
        plot_cytokine(df_neg, cytokine, axes[0], title_suffix=" (IL-1B-)")
        plot_cytokine(df_pos, cytokine, axes[1], title_suffix=" (IL-1B+)")
        fig.tight_layout()
        save_figure(
            fig,
            OUTPUT_DIR / "analytes" / safe_filename(cytokine)
            / "condition_boxplots.png",
        )


def plot_heatmaps(
    df: pd.DataFrame,
    analytes: list[str],
    complete_analytes: list[str],
) -> None:
    """Save the notebook heatmap and clustermap as static PNG files."""
    indexed = df.set_index(["condition", "bio_rep", "saw_il1b"])[analytes]
    df_z = (indexed - indexed.mean()) / indexed.std()
    plot_df = df_z.T  # transpose: analytes as rows, samples as columns

    fig, ax = plt.subplots(figsize=(14, 9))
    sns.heatmap(
        plot_df, cmap="vlag", center=0, annot=False,
        cbar_kws={"label": "z-scored concentration"}, ax=ax,
    )
    # pull condition labels directly from the MultiIndex level
    conditions = df_z.index.get_level_values("condition")
    unique_conditions = conditions.unique()
    condition_labels = [CONDITION_MAP.get(condition) for condition in unique_conditions]
    # group sizes may vary — compute boundaries dynamically rather than assuming a fixed size
    group_sizes = conditions.value_counts(sort=False).reindex(unique_conditions)
    boundaries = np.concatenate([[0], group_sizes.cumsum().values])
    tick_positions = (boundaries[:-1] + boundaries[1:]) / 2
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(condition_labels, rotation=45, ha="right", fontsize=9)
    for boundary in boundaries[1:-1]:
        ax.axvline(boundary, color="white", linewidth=1.5)
    ax.set_xlabel("condition")
    ax.set_ylabel("analyte")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "heatmaps" / "analyte_heatmap.png")

    # SEANS ATTEMPT AT MAKING CLUSTERMAP WORK NICELY. IT DOESNT
    # pheatmap() in r much easier imo. matplotlib so scare
    # Generate a list/Series of RGBA colors matching each row
    # Hierarchical clustering cannot accept missing values. As in the previous
    # workflow, cluster only analytes that are complete across all samples.
    cluster_data = df_z[complete_analytes].T
    cluster = sns.clustermap(cluster_data, cmap="vlag", center=0, figsize=(12, 10))
    cluster.fig.savefig(
        OUTPUT_DIR / "heatmaps" / "analyte_clustermap.png",
        dpi=300, bbox_inches="tight", facecolor="white",
    )
    plt.close(cluster.fig)


def plot_cell_counts(df: pd.DataFrame) -> None:
    """Save cell-count summaries from the merged imaging measurements."""
    melted = df.melt(
        id_vars=["condition", "saw_il1b"],
        value_vars=["num ESCs", "num macrophages"],
        var_name="cell_type", value_name="count",
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
    for ax, il1b_value, title in zip(axes, [False, True], ["IL-1B-", "IL-1B+"]):
        sns.boxplot(
            data=melted[melted["saw_il1b"] == il1b_value],
            x="condition", y="count", hue="cell_type", ax=ax,
        )
        ax.set_title(title)
        if ax is not axes[-1] and ax.legend_ is not None:
            ax.legend_.remove()
    axes[-1].legend(title="cell_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.suptitle("Cell Counts by Condition")
    fig.tight_layout()
    save_figure(fig, OUTPUT_DIR / "cell_counts" / "counts_by_condition.png")


def differential_expression(df: pd.DataFrame, analytes: list[str]) -> pd.DataFrame:
    """Compute and plot +IL-1B versus -IL-1B log2 fold changes."""
    melted = df.melt(
        id_vars=["condition", "bio_rep", "saw_il1b"],
        value_vars=analytes, var_name="cytokine", value_name="value",
    )
    # log2-transform each replicate (add pseudocount if you ever hit 0/negative values)
    if (melted["value"] <= 0).any():
        raise ValueError("Differential expression requires positive concentrations.")
    melted["log2fc"] = np.log2(melted["value"])

    # per condition x cytokine x saw_il1b: mean and SE across tech reps (skips NAs automatically)
    stats = (
        melted.groupby(["condition", "cytokine", "saw_il1b"])["log2fc"]
        .agg(mean="mean", std="std", n="count").reset_index()
    )
    stats["se"] = stats["std"] / np.sqrt(stats["n"])

    # pivot so +/- IL1B sit side by side, then compute logFC and propagated SE
    pivoted = stats.pivot(
        index=["condition", "cytokine"], columns="saw_il1b",
        values=["mean", "se"],
    )
    logfc_df = pd.DataFrame({
        "logFC": pivoted[("mean", True)] - pivoted[("mean", False)],
        "se": np.sqrt(pivoted[("se", True)] ** 2 + pivoted[("se", False)] ** 2),
    }).reset_index()
    logfc_df.to_csv(OUTPUT_DIR / "differential_expression.csv", index=False)

    for cond in sorted(melted["condition"].unique()):
        fig = plot_condition(logfc_df, cond)
        save_figure(
            fig, OUTPUT_DIR / "differential_expression"
            / f"condition_{safe_filename(cond)}.png",
        )
    return logfc_df


def inflammatory_readouts(df: pd.DataFrame):

    # take well avgs
    df['saw_il1b'] = df['bio_rep'].isin(['B', 'C', 'D']).astype(bool)
    df = df.groupby(['bio_rep', 'condition','cell_type']).mean().reset_index()

    readouts = {'phospho c-Jun': 'channel_4_nuclear_mean', 'NFkB n:c': 'channel_3_nc_ratio'}
    cell_types = ['macrophage', 'ESC']

    for label, col in readouts.items():
        fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        for ax, cell_type in zip(axs, cell_types):
            sub_df = df[df['cell_type'] == cell_type]
            sns.boxplot(sub_df, x="condition", y=col, hue="saw_il1b", ax=ax)
            ax.set_title(f"{cell_type} {label}")
            ax.set_ylabel(f"mean {label}")

        fig.tight_layout()
        save_figure(
            fig,
            OUTPUT_DIR / "inflammation" / safe_filename(label)
            / "condition_boxplots.png",
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df = pd.read_csv(DATA_DIR / "merged_data.csv")
    complete_analytes = complete_analyte_columns(merged_df)
    if len(complete_analytes) < 2:
        raise ValueError("PCA/clustering require at least two complete analytes.")

    # example of using plot_cytokine():
    # plot_cytokine(df, 'TNFa')
    plot_pca_outputs(merged_df, complete_analytes)
    plot_analyte_boxplots(merged_df, ANALYTE_COLUMNS)
    plot_heatmaps(merged_df, ANALYTE_COLUMNS, complete_analytes)
    plot_cell_counts(merged_df)
    differential_expression(merged_df, ANALYTE_COLUMNS)

    per_cell_df = pd.read_csv(DATA_DIR / "nucleus_data.csv")
    inflammatory_readouts(per_cell_df)

    print(f"Saved figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
