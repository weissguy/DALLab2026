# GOAL:  make interactive regression plots using plotly


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import ipywidgets as widgets
from IPython.display import display, clear_output

# load data:
df = pd.read_csv('data/luminex_imaging_counts_merged.csv')
analytes = df.columns[3:-3]


n_graphs = len(analytes)
current = {'i': 0}

output = widgets.Output()
btn_prev = widgets.Button(description='◀ Previous')
btn_next = widgets.Button(description='Next ▶')
label = widgets.Label()

def render():
    label.value = f"Graph {current['i']+1} / {n_graphs}"
    with output:
        clear_output(wait=True)
        cytokine = analytes[current['i']]

        x_col, y_col = 'num ESCs', 'num macrophages'
        sub = df[[x_col, y_col, cytokine]].dropna()

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
        fig = px.scatter_3d(sub, x=x_col, y=y_col, z=cytokine)
        fig.update_traces(marker=dict(size=4))

        # add the regression surface
        fig.add_trace(go.Surface(
            x=x_range, y=y_range, z=z_grid,
            opacity=0.5, showscale=False, colorscale='Reds',
            name='Linear fit'
        ))

        fig.update_layout(
            title=f"{cytokine}  (R² = {r2:.3f})",
            scene=dict(
                xaxis_title=x_col,
                yaxis_title=y_col,
                zaxis_title=cytokine
            )
        )
        fig.show()

def on_next(b):
    current['i'] = min(current['i'] + 1, n_graphs - 1)
    render()

def on_prev(b):
    current['i'] = max(current['i'] - 1, 0)
    render()

btn_next.on_click(on_next)
btn_prev.on_click(on_prev)

display(widgets.HBox([btn_prev, btn_next]), label, output)
render()


# alt:static version

sns.scatterplot(x = 'num macrophages', y = 'num ESCs', hue = 'TNFa', data = df)

# ok im not in love with this LOL but maybe a way to save those out or smth...

# or just make 


# POTENTIALLY INTERESTING:

# fit linear models for each analyte, report R^2: how much variation in that analyte is captured by 
# change in cell type proportions? (how) does this vary for ESC vs MAC driven cytokines?


# ALTERNATIVELY (maybbe i like this more): fit TWO linear models for each analyte, one using num ESCs as predictor one using 
# num macs as a predictor. see which celltype number is useful in predicting levels in either; we can maybe extrapolate this
# as cell type specificity of that analyte (e.g. if num of ESCs is all you need to know to predict level of CCL5, then probably
# just the ESCs are secreting CCL5)

# does this change with IL1B stim vs unstim?
