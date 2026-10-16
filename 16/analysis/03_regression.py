"""03: Fit cell-count regressions and save one static figure per analyte."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.my_funcs import (
    DATA_DIR, FIGURE_DIR, ANALYTE_COLUMNS, safe_filename, save_figure,
)

OUTPUT_DIR = FIGURE_DIR / "03_regression"
GROUP_COLORS = {True: "tab:orange", False: "tab:blue"}
GROUP_TITLES = {True: "IL-1B+", False: "IL-1B-"}
CELL_COUNT_PREDICTORS = ("num ESCs", "num macrophages")


def cell_type_regression(
    df: pd.DataFrame, analytes: list[str]
) -> tuple[plt.Figure, pd.DataFrame]:
    """Fit each analyte on each cell count alone, pooling IL-1B groups.

    R² measures in-sample fit, not validated predictive performance. These
    separate models do not estimate effects adjusted for the other cell type.
    Slopes are tested with OLS; q-values use BH correction across all fits.
    """
    rows = []
    for cytokine in analytes:
        # fit a model for num ESCs and num macrophages independently
        for predictor in CELL_COUNT_PREDICTORS:
            sub = df[[predictor, cytokine]].replace([np.inf, -np.inf], np.nan).dropna()
            row = {
                "analyte": cytokine, "predictor": predictor,
                "model": f"{cytokine} ~ {predictor}", "n": len(sub),
                "r2": np.nan, "intercept": np.nan, "weight": np.nan,
                "standardized_weight": np.nan, "p_value": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan,
                "status": "insufficient data or constant values",
            }
            if len(sub) >= 3 and sub[predictor].nunique() > 1 and sub[cytokine].nunique() > 1:
                x = sub[predictor].to_numpy()
                y = sub[cytokine].to_numpy()
                model = sm.OLS(y, sm.add_constant(x)).fit()
                weight = float(model.params[1])
                scale = x.std(ddof=0) / y.std(ddof=0)
                interval = model.conf_int(alpha=0.05)[1]
                row.update({
                    "r2": float(model.rsquared),
                    "intercept": float(model.params[0]), "weight": weight,
                    "standardized_weight": weight * x.std(ddof=0) / y.std(ddof=0),
                    "p_value": float(model.pvalues[1]), "status": "ok",
                    "ci_lower": float(interval[0] * scale),
                    "ci_upper": float(interval[1] * scale),
                })
            rows.append(row)

    results = pd.DataFrame(rows)
    results["q_value"] = np.nan
    valid = results["p_value"].notna()
    if valid.any():
        results.loc[valid, "q_value"] = multipletests(
            results.loc[valid, "p_value"], method="fdr_bh"
        )[1]
    results["significant_fdr_0.05"] = results["q_value"] < 0.05

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, predictor, color in zip(axes, CELL_COUNT_PREDICTORS, ["tab:blue", "tab:orange"]):
        subset = results[(results["predictor"] == predictor) & results["q_value"].notna()]
        significance = -np.log10(subset["q_value"].clip(lower=np.finfo(float).tiny))
        ax.scatter(subset["standardized_weight"], significance, color=color)
        for (_, row), height in zip(subset.iterrows(), significance):
            ax.annotate(row["analyte"], (row["standardized_weight"], height),
                        xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", label="FDR = 0.05")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Standardized slope")
        ax.set_title(predictor)
        ax.legend()
    axes[0].set_ylabel("−log10(BH-adjusted p-value)")
    fig.suptitle("Single cell-count regressions: effect and significance")
    fig.tight_layout()
    return fig, results


def plot_cell_type_r2(results: pd.DataFrame) -> plt.Figure:
    """Compare single-predictor standardized weights (95% CIs) and R²."""
    analytes = results["analyte"].drop_duplicates().tolist()
    fig, (effect_ax, fit_ax) = plt.subplots(
        2, 1, figsize=(max(12, len(analytes) * 0.65), 12),
    )
    positions = np.arange(len(analytes))
    width = 0.38
    for offset, predictor, color in zip(
        [-width / 2, width / 2], CELL_COUNT_PREDICTORS, ["tab:blue", "tab:orange"]
    ):
        subset = results[results["predictor"] == predictor].set_index("analyte").reindex(analytes)
        weights = subset["standardized_weight"].to_numpy()
        effect_ax.bar(
            positions + offset, weights, width, label=predictor, color=color,
            yerr=np.array([weights - subset["ci_lower"].to_numpy(),
                           subset["ci_upper"].to_numpy() - weights]),
            capsize=3, error_kw={"elinewidth": 1, "capthick": 1},
        )
        bars = fit_ax.bar(positions + offset, subset["r2"], width, label=predictor, color=color)
        for bar, (_, row) in zip(bars, subset.iterrows()):
            if row["significant_fdr_0.05"]:
                fit_ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                            "*", ha="center", va="bottom")
    effect_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    effect_ax.set_ylabel("Standardized coefficient (95% CI)")
    effect_ax.set_title("Each cell count as a sole predictor: weights")
    fit_ax.set_ylim(0, 1.08)
    fit_ax.set_ylabel("R² (in-sample variance explained)")
    fit_ax.set_title("Each cell count as a sole predictor; * FDR < 0.05")
    for ax in (effect_ax, fit_ax):
        ax.set_xticks(positions, analytes, rotation=60, ha="right")
        ax.set_xlabel("Analyte")
        ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


def multivariable_regression(
    df: pd.DataFrame, analytes: list[str], output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """Fit all analytes on ESC and macrophage counts and save the summary."""
    rows = []
    for cytokine in analytes:
        sub = df[[*CELL_COUNT_PREDICTORS, cytokine]].dropna()
        if len(sub) < 3:
            raise ValueError(
                f"{cytokine} needs at least three complete samples for regression."
            )
        X = sub[list(CELL_COUNT_PREDICTORS)].to_numpy()
        y = sub[cytokine].to_numpy()
        design = sm.add_constant(X, has_constant="add")
        reg = sm.OLS(y, design).fit()
        row = {
            "analyte": cytokine,
            "r2": reg.rsquared,
            "adjusted_r2": reg.rsquared_adj,
            "intercept": reg.params[0],
            "esc_coefficient": reg.params[1],
            "macrophage_coefficient": reg.params[2],
            "n": len(sub),
            "effects_estimable": np.linalg.matrix_rank(design) == 3 and reg.df_resid > 0,
        }
        intervals = reg.conf_int(alpha=0.05)
        for i, prefix in enumerate(["esc", "macrophage"]):
            scale = X[:, i].std(ddof=0) / y.std(ddof=0) if y.std(ddof=0) > 0 else np.nan
            for suffix, value in [
                ("standardized_coefficient", reg.params[i + 1]),
                ("ci_lower", intervals[i + 1, 0]),
                ("ci_upper", intervals[i + 1, 1]),
            ]:
                row[f"{prefix}_{suffix}"] = value * scale if row["effects_estimable"] else np.nan
            row[f"{prefix}_p_value"] = (
                float(reg.pvalues[i + 1])
                if row["effects_estimable"] and np.isfinite(scale) else np.nan
            )
        rows.append(row)
    results = pd.DataFrame(rows)
    # One BH family covering both coefficients across all analytes.
    p_columns = ["esc_p_value", "macrophage_p_value"]
    p_values = results[p_columns].to_numpy()
    valid = np.isfinite(p_values)
    q_values = np.full_like(p_values, np.nan)
    if valid.any():
        q_values[valid] = multipletests(p_values[valid], method="fdr_bh")[1]
    for i, prefix in enumerate(["esc", "macrophage"]):
        results[f"{prefix}_q_value"] = q_values[:, i]
        results[f"{prefix}_significant_fdr_0.05"] = q_values[:, i] < 0.05
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "multivariable_regression_summary.csv", index=False)
    fig = plot_multivariable_summary(results)
    save_figure(fig, output_dir / "multivariable" / "coefficients_and_fit.png")
    fig = plot_multivariable_volcano(results)
    save_figure(fig, output_dir / "multivariable" / "linear_regression_volcano_plot.png")
    return results


def plot_multivariable_volcano(results: pd.DataFrame) -> plt.Figure:
    """Plot conditional coefficients versus BH-adjusted slope significance."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    for ax, prefix, predictor, color in zip(
        axes, ["esc", "macrophage"], CELL_COUNT_PREDICTORS, ["tab:blue", "tab:orange"]
    ):
        valid = results[f"{prefix}_q_value"].notna()
        subset = results.loc[valid]
        effect = subset[f"{prefix}_standardized_coefficient"]
        height = -np.log10(subset[f"{prefix}_q_value"].clip(lower=np.finfo(float).tiny))
        ax.scatter(effect, height, color=color)
        for analyte, x, y in zip(subset["analyte"], effect, height):
            ax.annotate(analyte, (x, y), xytext=(4, 3),
                        textcoords="offset points", fontsize=8)
        ax.axhline(-np.log10(0.05), color="gray", linestyle="--", label="FDR = 0.05")
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Standardized coefficient (adjusted for other cell count)")
        ax.set_title(predictor)
        ax.legend()
    axes[0].set_ylabel("−log10(BH-adjusted p-value)")
    fig.suptitle("Multivariable cell-count regressions: effect and significance (pooled IL-1B)")
    fig.tight_layout()
    return fig


def plot_multivariable_summary(results: pd.DataFrame) -> plt.Figure:
    """Plot paired coefficient bars with 95% CIs and combined-model R²."""
    fig, (effect_ax, fit_ax) = plt.subplots(
        2, 1, figsize=(max(12, len(results) * 0.65), 12),
    )
    positions = np.arange(len(results))
    width = 0.38
    for prefix, label, color, offset in [
        ("esc", "num ESCs", "tab:blue", -width / 2),
        ("macrophage", "num macrophages", "tab:orange", width / 2),
    ]:
        coefficient = results[f"{prefix}_standardized_coefficient"].to_numpy()
        lower = results[f"{prefix}_ci_lower"].to_numpy()
        upper = results[f"{prefix}_ci_upper"].to_numpy()
        effect_ax.bar(
            positions + offset, coefficient, width,
            yerr=np.array([coefficient - lower, upper - coefficient]),
            color=color, capsize=3, label=label,
            error_kw={"elinewidth": 1, "capthick": 1},
        )
    effect_ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    effect_ax.set_ylabel("Standardized coefficient (95% CI)")
    effect_ax.set_title("Each cell count adjusted for the other")
    effect_ax.legend(loc="upper right")
    fit_ax.bar(positions, results["r2"], width=2 * width,
               label="ESCs + macrophages", color="tab:blue")
    fit_ax.set_ylim(0, 1.08)
    fit_ax.set_ylabel("R² (in-sample variance explained)")
    fit_ax.set_title("Both cell counts as predictors")
    fit_ax.legend(loc="upper right")
    for ax in (effect_ax, fit_ax):
        ax.set_xticks(positions, results["analyte"], rotation=60, ha="right")
        ax.set_xlabel("Analyte")
    fig.tight_layout()
    return fig


def ratio_regression(
    df: pd.DataFrame, analytes: list[str], output_dir: Path = OUTPUT_DIR
) -> pd.DataFrame:
    """Save ratio-regression figures for all analytes and a combined summary."""
    df = df.copy()
    denominator = df["num macrophages"].replace(0, np.nan)
    df["ESC:macrophage"] = df["num ESCs"] / denominator
    rows: list[dict[str, float | str | bool]] = []
    for cytokine in analytes:
        x_col = "ESC:macrophage"
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

        for ax, (il1b_val, color) in zip(axes, GROUP_COLORS.items()):
            sub = df[df["saw_il1b"] == il1b_val][[x_col, cytokine]].dropna()
            ax.scatter(sub[x_col], sub[cytokine], color=color, alpha=0.7)

            if len(sub) >= 3 and sub[x_col].nunique() > 1:
                X = sub[x_col].to_numpy()
                y = sub[cytokine].to_numpy()
                X_design = sm.add_constant(X)  # adds intercept column
                model = sm.OLS(y, X_design).fit()
                x_line = np.linspace(X.min(), X.max(), 100)
                x_line_design = sm.add_constant(x_line)
                pred = model.get_prediction(x_line_design)
                pred_summary = pred.summary_frame(alpha=0.32)  # ~1 SE band (≈68% CI)
                ax.plot(x_line, pred_summary["mean"], color=color, linewidth=2)
                ax.fill_between(
                    x_line, pred_summary["mean_ci_lower"], pred_summary["mean_ci_upper"],
                    color=color, alpha=0.2,
                )
                ax.set_title(f"{GROUP_TITLES[il1b_val]} (R² = {model.rsquared:.3f})")
                rows.append({
                    "analyte": cytokine,
                    "saw_il1b": il1b_val,
                    "r2": model.rsquared,
                    "intercept": model.params[0],
                    "ratio_coefficient": model.params[1],
                    "n": len(sub),
                })
            else:
                ax.set_title(f"{GROUP_TITLES[il1b_val]} (insufficient data)")
            ax.set_xlabel(x_col)

        axes[0].set_ylabel(cytokine)
        fig.suptitle(f"{cytokine} vs {x_col}")
        fig.tight_layout()
        save_figure(fig, output_dir / "analytes" / safe_filename(cytokine) / "esc_macrophage_ratio.png")
    results = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "ratio_regression_summary.csv", index=False)
    return results


def inflammation_regression(df: pd.DataFrame):
    pass


def plot_cell_relationship(
    df: pd.DataFrame, analytes: list[str], output_dir: Path = OUTPUT_DIR
) -> None:
    """Save all ESC-versus-macrophage figures colored by analyte level."""
    for cytokine in analytes:
        fig, ax = plt.subplots(figsize=(8, 6))
        scatter = ax.scatter(
            df["num macrophages"], df["num ESCs"], c=df[cytokine], cmap="viridis",
        )
        ax.set_xlabel("num macrophages")
        ax.set_ylabel("num ESCs")
        ax.set_title(f"Cell counts colored by {cytokine}")
        fig.colorbar(scatter, ax=ax, label=cytokine)
        fig.tight_layout()
        save_figure(fig, output_dir / "analytes" / safe_filename(cytokine) / "cell_count_2d.png")


def main() -> None:
    # load data:
    df = pd.read_csv(DATA_DIR / "merged_data.csv")
    analytes = ANALYTE_COLUMNS
    if not analytes:
        raise ValueError("No analyte columns were found in the merged CSV.")

    multivariable_regression(df, analytes)
    plot_cell_relationship(df, analytes)
    ratio_regression(df, analytes)

    fig, independent_results = cell_type_regression(df, analytes)
    save_figure(fig, OUTPUT_DIR / 'single_variable' / 'linear_regression_volcano_plot.png')

    fig = plot_cell_type_r2(independent_results)
    save_figure(fig, OUTPUT_DIR / 'single_variable' / 'linear_regression_r2_bars.png')
    independent_results.to_csv(
        OUTPUT_DIR / "independent_cell_count_regression_summary.csv", index=False
    )

    print(f"Saved figures and summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
