"""01: Combine the Luminex measurements with imaging-derived cell counts."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / "data"

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from functools import reduce

from utils.my_funcs import RF_FEATURES



################################################
# Processing Luminex data ---
################################################


def prepare_luminex(path: Path) -> pd.DataFrame:
    """Aggregate technical measurements and pivot analytes into columns."""
    df = pd.read_csv(path)
    df["analyte"] = df["analyte"].str.split("_").str[0]

    # aggregate by condition
    df = (
        df.dropna(subset="concentration")
        .groupby(["condition", "bio_rep", "analyte", "saw_il1b"], as_index=False)[
            "concentration"
        ]
        .mean()
    )

    # pivot such that each row represents a condition and each column represents an analyte
    df = df.pivot_table(
        index=["condition", "bio_rep", "saw_il1b"],
        columns="analyte",
        values="concentration",
    ).reset_index()
    df.columns.name = None

    # drop analytes that are mainly missing
    df = df.drop(["MMP7"], axis=1)

    return df


################################################
# Adding in cell counts ---
################################################


def prepare_cell_counts(path: Path) -> pd.DataFrame:
    """Derive merge keys from imaging well IDs."""
    cell_counts = pd.read_csv(path, index_col=0)

    # separate bio rep and condition from well_id
    cell_counts["bio_rep"] = cell_counts["well_id"].str.extract(r"([A-Za-z]+)")
    cell_counts["condition"] = cell_counts["well_id"].str.extract(r"(\d+)").astype(int)
    return cell_counts.drop(columns="well_id")


################################################
# Adding in cell counts ---
################################################


def classify_cells(df: pd.DataFrame, verbose=False) -> pd.Series:
    """Returns a Series of cell type classifications for df."""
    # classify cells as ESC or macrophage
    df["cell_type"] = np.select(
        [
            ((df["condition"] == 2) | (df["condition"] == 3)),
            ((df["condition"] == 8) | (df["condition"] == 9))
        ],
        [
            "macrophage",
            "ESC"
        ],
        default=None
    )

    mask = ~df["cell_type"].isna()
    sub_df = df[mask].copy()
    X = sub_df[RF_FEATURES].to_numpy()
    y = sub_df["cell_type"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    rf = RandomForestClassifier().fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    if verbose:
        print(classification_report(y_test, y_pred))

    return rf.predict(df[RF_FEATURES].to_numpy())


def compute_well_avg(df: pd.DataFrame, keys: list, cell_type_specific_cols: dict) -> pd.DataFrame:
    """
    Returns a df where measurements are averaged. Each row corresponds to a 
    unique combination of keys. For cell_type_specific_cols, means are computed
    indepenently for each cell type and reported as separate columns.

    - df: DataFrame with per-cell measurements
    - keys: list of column names that will serve as row unique identifiers
    - cell_type_specific_cols: dict {label: column} of columns that should be
      aggregated and reported independently by cell type
    """
    # 1. pooled mean of every other measurement
    pooled = (df.drop(columns=["cell_type", *cell_type_specific_cols.values()])
                .groupby(keys)
                .mean(numeric_only=True))

    # 2. per-cell-type mean of the two split measurements
    by_type = (df.groupby(keys + ["cell_type"])[list(cell_type_specific_cols.values())]
                .mean()
                .unstack("cell_type"))

    # flatten the (column, cell_type) MultiIndex into e.g. "phospho c-Jun macrophage"
    inv = {v: k for k, v in cell_type_specific_cols.items()}
    by_type.columns = [f"{inv[col]} {ct}" for col, ct in by_type.columns]

    return pooled.join(by_type).reset_index()


def prepare_cellpose(path: Path) -> pd.DataFrame:
    """Return well-average measurements."""
    cellpose = pd.read_csv(path, index_col=0)

    # separate bio rep and condition from well_id
    cellpose["bio_rep"] = cellpose["well_id"].str.extract(r"([A-Za-z]+)")
    cellpose["condition"] = cellpose["well_id"].str.extract(r"(\d+)").astype(int)
    cellpose = cellpose.drop(columns=["well_id"])

    # classify cell types with RF
    cellpose["cell_type"] = classify_cells(cellpose)

    # average measurements per well
    keys = ["bio_rep", "condition"]
    split_cols = {"phospho c-Jun": "channel_4_nuclear_mean",
                  "NFkB n:c": "channel_3_nc_ratio"}
    well_avg_df = compute_well_avg(cellpose, keys, split_cols)

    return well_avg_df



def main():

    luminex = prepare_luminex(DATA_DIR / "luminex_wrangled.csv")
    cell_counts = prepare_cell_counts(DATA_DIR / "cell_counts.csv")
    cellpose = prepare_cellpose(DATA_DIR / "cellpose_measurements.csv")

    # MERGE
    dfs = [luminex, cell_counts, cellpose]
    merged_df = reduce(
        lambda left, right: pd.merge(
            left,
            right,
            on=["condition", "bio_rep"],
            how="inner",
            validate="many_to_one",
        ),
        dfs
    )

    # SAVE
    merged_df.to_csv(DATA_DIR / "merged_data.csv", index=False) # one row per well
    print(f"Saved {len(merged_df)} rows to {DATA_DIR / "merged_data.csv"}")

    #cellpose_per_nuc.to_csv(DATA_DIR / "nucleus_data.csv", index=False) # one row per cell
    #print(f"Saved {len(cellpose_per_nuc)} rows to {DATA_DIR / "nucleus_data.csv"}")



if __name__ == "__main__":
    main()
