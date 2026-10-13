# Luminex analysis

Run the analysis from any working directory in numbered order:

```bash
python analysis/01_combine_data.py
python analysis/02_initial_viz.py
python analysis/03_regression.py
```

The scripts use paths relative to this project rather than changing the current
working directory. Step 01 creates `data/luminex_imaging_counts_merged.csv`.
Steps 02 and 03 save static PNG files and tabular summaries under `figures/`.
Per-analyte outputs are grouped as `figures/<analysis>/analytes/<analyte>/`.
Step 03 also writes `independent_cell_count_regression_summary.csv`, containing
separate cytokine-on-ESC and cytokine-on-macrophage R² values and weights.

The merged CSV retains both IL-1B groups and analytes with partial missingness.
Per-analyte plots and regressions use every available measurement. PCA and the
cluster map retain the earlier complete-case analyte policy because those
methods cannot operate directly on missing values.

Install the analysis dependencies with:

```bash
python -m pip install -r requirements.txt
```

See `MERGE_REVIEW.md` for notebook/script differences that were resolved in
favor of `luminex.ipynb` and should receive a human review.
