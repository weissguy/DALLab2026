"""03: Fit cell-count regressions and save one static figure per analyte."""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402
import statsmodels.api as sm  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.my_funcs import (  # noqa: E402
    DATA_DIR, FIGURE_DIR, ANALYTE_COLUMNS, safe_filename, save_figure,
)


# GOAL: make interactive regression plots using plotly
# The interactive notebook browser has been replaced with deterministic PNG export.
OUTPUT_DIR = FIGURE_DIR / "03_regression"
GROUP_COLORS = {True: "tab:orange", False: "tab:blue"}
GROUP_TITLES = {True: "IL-1B+", False: "IL-1B-"}
CELL_COUNT_PREDICTORS = ("num ESCs", "num macrophages")


def cell_type_regression(
    df: pd.DataFrame, analytes: list[str]
) -> pd.DataFrame:
    """Fit cytokine ~ cell count separately for ESCs and macrophages.

    The raw weight is the change in cytokine concentration per additional cell.
    The standardized weight is also included to make effects more comparable
    across analytes with different concentration scales.
    """
    rows = []
    for cytokine in analytes:
        for predictor in CELL_COUNT_PREDICTORS:
            sub = df[[predictor, cytokine]].dropna()
            if len(sub) < 2:
                continue
            X = sub[[predictor]].to_numpy()
            y = sub[cytokine].to_numpy()
            model = LinearRegression().fit(X, y)
            predictions = model.predict(X)
            weight = float(model.coef_[0])
            y_std = float(y.std(ddof=0))
            standardized_weight = (
                weight * float(X[:, 0].std(ddof=0)) / y_std
                if y_std > 0
                else np.nan
            )
            rows.append({
                "analyte": cytokine,
                "predictor": predictor,
                "model": f"{cytokine} ~ {predictor}",
                "r2": r2_score(y, predictions),
                "intercept": float(model.intercept_),
                "weight": weight,
                "standardized_weight": standardized_weight,
                "n": len(sub),
            })
    return pd.DataFrame(rows)


def multivariable_regression(
    df: pd.DataFrame, cytokine: str
) -> tuple[plt.Figure, dict[str, float]]:
    """Plot cytokine ~ ESC count + macrophage count as a static 3D surface."""
    x_col, y_col = "num ESCs", "num macrophages"
    sub = df[[x_col, y_col, cytokine]].dropna()
    if len(sub) < 3:
        raise ValueError(
            f"{cytokine} needs at least three complete samples for regression."
        )

    # fit linear regression: cytokine ~ num_ESCs + num_macrophages
    X = sub[[x_col, y_col]].to_numpy()
    z = sub[cytokine].to_numpy()
    reg = LinearRegression().fit(X, z)
    r2 = r2_score(z, reg.predict(X))

    # build a grid over the observed x/y range for the fitted plane
    x_range = np.linspace(sub[x_col].min(), sub[x_col].max(), 20)
    y_range = np.linspace(sub[y_col].min(), sub[y_col].max(), 20)
    x_grid, y_grid = np.meshgrid(x_range, y_range)
    z_grid = reg.predict(
        np.column_stack([x_grid.ravel(), y_grid.ravel()])
    ).reshape(x_grid.shape)

    # scatter of the actual points
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(sub[x_col], sub[y_col], sub[cytokine], s=24, alpha=0.8)
    # add the regression surface
    ax.plot_surface(x_grid, y_grid, z_grid, cmap="Reds", alpha=0.45)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(cytokine)
    ax.set_title(f"{cytokine} (R² = {r2:.3f})")
    fig.tight_layout()
    return fig, {
        "analyte": cytokine,
        "r2": r2,
        "intercept": reg.intercept_,
        "esc_coefficient": reg.coef_[0],
        "macrophage_coefficient": reg.coef_[1],
        "n": len(sub),
    }


def ratio_regression(
    df: pd.DataFrame, cytokine: str
) -> tuple[plt.Figure, list[dict[str, float | str | bool]]]:
    """Regress one analyte on the ESC:macrophage ratio within IL-1B groups."""
    x_col = "ESC:macrophage"
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    rows: list[dict[str, float | str | bool]] = []

    for ax, (il1b_val, color) in zip(axes, GROUP_COLORS.items()):
        sub = df[df["saw_il1b"] == il1b_val][[x_col, cytokine]].dropna()
        ax.scatter(sub[x_col], sub[cytokine], color=color, alpha=0.7)

        if len(sub) >= 2:
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
    return fig, rows


def plot_cell_relationship(df: pd.DataFrame, cytokine: str) -> None:
    """Save the notebook's static ESC-versus-macrophage diagnostic."""
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["num macrophages"], df["num ESCs"], c=df[cytokine], cmap="viridis",
    )
    ax.set_xlabel("num macrophages")
    ax.set_ylabel("num ESCs")
    ax.set_title(f"Cell counts colored by {cytokine}")
    fig.colorbar(scatter, ax=ax, label=cytokine)
    fig.tight_layout()
    return fig


def main() -> None:
    # load data:
    df = pd.read_csv(DATA_DIR / "merged_data.csv")
    analytes = ANALYTE_COLUMNS
    if not analytes:
        raise ValueError("No analyte columns were found in the merged CSV.")

    # regression cytokine ~ ESC/macrophage, within +IL-1B and -IL-1B groups
    denominator = df["num macrophages"].replace(0, np.nan) # prevent / 0
    df["ESC:macrophage"] = df["num ESCs"] / denominator

    # TODO: clean up multivariable regression, model cytokine ~ ESC + Mac + ESC:Mac
    multivariable_rows = []
    ratio_rows = []
    for cytokine in analytes:
        analyte_dir = OUTPUT_DIR / "analytes" / safe_filename(cytokine)
        fig, summary = multivariable_regression(df, cytokine)
        save_figure(fig, analyte_dir / "cell_count_plane.png")
        multivariable_rows.append(summary)

        fig = plot_cell_relationship(df, cytokine)
        save_figure(fig, analyte_dir / "cell_count_2d.png")

        fig, summaries = ratio_regression(df, cytokine)
        save_figure(fig, analyte_dir / "esc_macrophage_ratio.png")
        ratio_rows.extend(summaries)

    pd.DataFrame(multivariable_rows).to_csv(
        OUTPUT_DIR / "multivariable_regression_summary.csv", index=False
    )
    pd.DataFrame(ratio_rows).to_csv(
        OUTPUT_DIR / "ratio_regression_summary.csv", index=False
    )
    independent_results = cell_type_regression(df, analytes)
    independent_results.to_csv(
        OUTPUT_DIR / "independent_cell_count_regression_summary.csv", index=False
    )

    print(f"Saved figures and summaries to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
