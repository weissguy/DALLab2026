"""Shared plotting and filesystem helpers for the Luminex analysis."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FIGURE_DIR = PROJECT_ROOT / "figures"

CONDITION_MAP = {
    2: "5k macs",
    3: "10k macs",
    4: "1k escs, 9k macs",
    5: "5k escs, 5k macs",
    6: "9k escs, 1k macs",
    7: "10k escs",
    8: "5k escs",
}

METADATA_COLUMNS = {
    "condition",
    "bio_rep",
    "saw_il1b",
    "num cells",
    "num ESCs",
    "num macrophages",
    "ESC:macrophage",
}

ANALYTE_COLUMNS = [
    "CCL2",
    "CCL3",
    "CCL4",
    "CCL5",
    "CXCL1",
    "CXCL10",
    "CXCL13",
    "CXCL5",
    "GCSF",
    "IL10",
    "IL1B",
    "IL1ra",
    "IL4",
    "IL6",
    "IL8",
    "MMP1",
    "MMP12",
    "MMP9",
    "SLPI",
    "TNFa",
    "VCAM1",
]

RF_FEATURES = ['nuclear_volume_voxels', 'cytoplasm_volume_voxels',
       'channel_1_nuclear_mean', 'channel_1_nuclear_max', 
       'channel_1_nuclear_std', 'channel_1_cytoplasm_mean', 
       'channel_1_cytoplasm_max', 'channel_1_cytoplasm_std', 'channel_1_nc_ratio',
       'channel_2_nuclear_mean', 'channel_2_nuclear_max',
       'channel_2_nuclear_std', 'channel_2_cytoplasm_mean',
       'channel_2_cytoplasm_max', 'channel_2_cytoplasm_std', 'channel_2_nc_ratio',
       'channel_3_nuclear_mean', 'channel_3_nuclear_max',
       'channel_3_nuclear_std', 'channel_3_cytoplasm_mean',
       'channel_3_cytoplasm_max', 'channel_3_cytoplasm_std', 'channel_3_nc_ratio',
       'channel_4_nuclear_mean', 'channel_4_nuclear_max',
       'channel_4_nuclear_std', 'channel_4_cytoplasm_mean',
       'channel_4_cytoplasm_max', 'channel_4_cytoplasm_std', 'channel_4_nc_ratio',
       'channel_5_nuclear_mean', 'channel_5_nuclear_max',
       'channel_5_nuclear_std', 'channel_5_cytoplasm_mean',
       'channel_5_cytoplasm_max', 'channel_5_cytoplasm_std', 'channel_5_nc_ratio',
       'centroid_z', 'axis_major_length', 'axis_minor_length', 'aspect_ratio',
       'nn_dist', 'nn5_mean_dist', 'num_neighbors_within_50_um', 'sphericity',
]


def complete_analyte_columns(df: pd.DataFrame) -> list[str]:
    """Return analytes with values for every sample (required by PCA/clustering)."""
    return [column for column in ANALYTE_COLUMNS if df[column].notna().all()]


def safe_filename(value: object) -> str:
    """Convert a label to a stable, filesystem-safe name."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def save_figure(fig: plt.Figure, path: Path, *, dpi: int = 300) -> None:
    """Save a publication-friendly PNG and close its Matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cytokine(
    df: pd.DataFrame, cytokine: str, ax: plt.Axes, title_suffix: str = ""
) -> None:
    """
    Plots cytokine traces

    Parameters:
    df (pd.DataFrame): df with observations as rows and cytokines as columns
    cytokine (str): cytokine to plot

    Returns:
    None
    """
    plot_data = df.dropna(subset=[cytokine])
    if plot_data.empty:
        ax.text(
            0.5,
            0.5,
            "No measurements",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
    else:
        sns.boxplot(data=plot_data, x="condition", y=cytokine, ax=ax)
        sns.stripplot(
            data=plot_data,
            x="condition",
            y=cytokine,
            color="black",
            alpha=0.55,
            size=3,
            ax=ax,
        )
    ax.set_title(f"{cytokine} Concentration by Condition{title_suffix}")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Concentration")
    ax.tick_params(axis="x", rotation=45)


def plot_condition(
    logfc_df: pd.DataFrame, cond: int, ax: plt.Axes | None = None
) -> plt.Figure:
    """Plot IL-1B differential expression for one condition."""
    sub = (
        logfc_df[logfc_df["condition"] == cond]
        .dropna(subset=["logFC"])
        .sort_values("logFC")
    )
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.figure
    ax.barh(
        sub["cytokine"],
        sub["logFC"],
        xerr=sub["se"].fillna(0),
        capsize=4,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log2FC (+IL1B / -IL1B)")
    ax.set_title(CONDITION_MAP.get(cond, f"Condition {cond}"))
    fig.tight_layout()
    return fig


def pca_analytes(
    df: pd.DataFrame, analytes: list[str], il1b_pos_only: bool = False,
    color_by: str = "num macrophages",
) -> tuple[PCA, plt.Figure]:
    """
    Plots PCA of all analytes.

    - df: df with observations as rows and cytokines as columns
    - color_by: "num macrophages" or "num ESCs"; shown as a continuous gradient
    - il1b_pos_only: ...
        - if True, subsets to only saw_il1b=True
        - else, plots both +IL1B and -IL1B (some conditions have many NA)

    Returns:
    - pca: PCA object fit
    """
    if color_by not in {"num macrophages", "num ESCs"}:
        raise ValueError(
            'color_by must be "num macrophages" or "num ESCs".'
        )
    plot_data = df.copy()
    if il1b_pos_only:
        plot_data = plot_data[plot_data["saw_il1b"]]

    # PCA cannot accept missing values, so use samples complete for all analytes.
    plot_data = plot_data.dropna(subset=analytes)
    color_values = pd.to_numeric(plot_data[color_by], errors="raise")
    if color_values.isna().any() or color_values.isin([float("inf"), -float("inf")]).any():
        raise ValueError(f"{color_by} must contain finite counts for every plotted sample.")
    X_scaled = StandardScaler().fit_transform(plot_data[analytes].values)

    pca = PCA(n_components=min(4, len(analytes), len(plot_data)))
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(
        {
            "color_value": color_values.values,
            "saw_il1b": plot_data["saw_il1b"].values,
            "PC1": X_pca[:, 0],
            "PC2": X_pca[:, 1],
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(
        vmin=pca_df["color_value"].min(), vmax=pca_df["color_value"].max()
    )

    if il1b_pos_only:
        ax.scatter(
            pca_df["PC1"],
            pca_df["PC2"],
            c=pca_df["color_value"],
            cmap=cmap,
            norm=norm,
        )
    else:
        marker_map = {True: "^", False: "o"}
        for il1b_val, marker in marker_map.items():
            subset = pca_df[pca_df["saw_il1b"] == il1b_val]
            ax.scatter(
                subset["PC1"],
                subset["PC2"],
                c=subset["color_value"],
                cmap=cmap,
                norm=norm,
                marker=marker,
                edgecolor="k",
                linewidth=0.3,
                label=f"saw_il1b={il1b_val}",
            )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.03
    )
    colorbar.set_label(color_by)

    if not il1b_pos_only:
        # build legend (shape)
        shape_handles = [
            Line2D(
                [0],
                [0],
                marker=marker,
                linestyle="",
                markerfacecolor="gray",
                markeredgecolor="k",
                markersize=8,
                label=f"saw_il1b={il1b_val}",
            )
            for il1b_val, marker in marker_map.items()
        ]
        ax.legend(
            handles=shape_handles,
            title="IL-1B",
            loc="best",
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title("PCA of Luminex Analytes")
    fig.tight_layout()
    return pca, fig


def pca_transposed_analytes(
    df: pd.DataFrame, analytes: list[str], il1b_pos_only: bool = False
) -> tuple[PCA, plt.Figure]:
    """Plot analytes using condition means as PCA features.

    Average replicates within each condition, then transpose to analyte rows
    and condition columns. Unless il1b_pos_only is True, the means include
    both IL-1B groups. Standardize each condition column across analytes.
    """
    plot_data = df[df["saw_il1b"]] if il1b_pos_only else df
    condition_means = plot_data.groupby("condition")[analytes].mean()
    matrix = condition_means.T.dropna()
    if min(matrix.shape) < 2:
        raise ValueError(
            "Transposed PCA requires at least two complete analytes and two conditions."
        )
    scaled = StandardScaler().fit_transform(matrix)
    pca = PCA(n_components=min(4, *matrix.shape))
    scores = pca.fit_transform(scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(scores[:, 0], scores[:, 1])
    for analyte, (pc1, pc2) in zip(matrix.index, scores[:, :2]):
        ax.annotate(
            analyte, (pc1, pc2), xytext=(4, 3),
            textcoords="offset points", fontsize=9,
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    suffix = " (IL-1B positive)" if il1b_pos_only else " (all samples)"
    ax.set_title(f"PCA of Analytes Across Condition Means{suffix}")
    fig.tight_layout()
    return pca, fig
