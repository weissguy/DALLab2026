# make some initial heatmaps!
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# can also import functions you've defined in a functions script:
from utils.my_funcs import plot_cytokine



df = pd.read_csv('data/luminex_imaging_counts_merged.csv')
df = df.drop(labels=['Unnamed: 0', 'num cells'], axis=1)
num_escs = df.pop('num ESCs')
num_macrophages = df.pop('num macrophages')
df = df.set_index(['condition', 'bio_rep'])
analytes = df.columns
df_z = (df - df.mean()) / df.std()

# example of using plot_cytokine():
plot_cytokine(df, 'TNFa')

plot_df = df_z.T  # transpose: analytes as rows, samples as columns

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(plot_df, cmap='vlag', center=0, annot=False, cbar_kws={'label': 'z-scored concentration'}, ax=ax)

# pull condition labels directly from the MultiIndex level
conditions = df_z.index.get_level_values('condition')
unique_conditions = conditions.unique()
condition_labels = [condition_map.get(condition) for condition in unique_conditions]

# group sizes may vary — compute boundaries dynamically rather than assuming a fixed size
group_sizes = conditions.value_counts().reindex(unique_conditions)
boundaries = np.concatenate([[0], group_sizes.cumsum().values])
tick_positions = (boundaries[:-1] + boundaries[1:]) / 2

ax.set_xticks([])
ax.set_xticks(tick_positions)
ax.set_xticklabels(condition_labels, rotation=45, ha='right', fontsize=9)

for b in boundaries[1:-1]:
    ax.axvline(b, color='white', linewidth=1.5)

ax.set_xlabel('condition')
ax.set_ylabel('analyte')
plt.tight_layout()
plt.show()



## SEANS ATTEMPT AT MAKING CLUSTERMAP WORK NICELY. IT DOESNT 
# pheatmap() in r much easier imo. matplotlib so scare

cmap = cm.get_cmap('viridis')  # Pick your continuous colormap
norm = mcolors.Normalize(vmin=num_macrophages.min(), vmax=num_macrophages.max())
mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

# Generate a list/Series of RGBA colors matching each row
row_colors = pd.Series(num_macrophages).map(lambda x: mapper.to_rgba(x))
row_colors.index = df_z.index


plot = sns.clustermap(df_z, row_colors=row_colors)

cax2 = plot.fig.add_axes([0.02, 0.2, 0.03, 0.4]) 
fig_cbar = plot.fig.colorbar(mapper, cax=cax2, orientation='vertical')
fig_cbar.set_label('Continuous Row Label')
plt.show()

sns.clustermap(df_z)
