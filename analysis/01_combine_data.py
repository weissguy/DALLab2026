### GOAL: combine luminex and imaging data
# save out as .csv

# load packages
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colormaps
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


################################################
# Processing Luminex data ---
################################################

condition_map = {
    2: '5k macs',
    3: '10k macs',
    4: '1k escs, 9k macs',
    5: '5k escs, 5k macs',
    6: '9k escs, 1k macs',
    7: '10k escs',
    8: '5k escs'
}

color_gradient = colormaps['RdYlBu']


df = pd.read_csv('data/luminex_wrangled.csv')
df['analyte'] = df['analyte'].str.split('_').str[0]

# aggregate by condition
df = df.dropna(subset='concentration').groupby(['condition', 'bio_rep', 'analyte', 'saw_il1b'])[['concentration']].mean().reset_index()

# pivot such that each row represents a condition and each column represents an analyte
df = df.pivot_table(index=['condition', 'bio_rep', 'saw_il1b'], columns='analyte', values='concentration').reset_index()

# only IL-1B+
df = df[df['saw_il1b']]
df = df.drop('saw_il1b', axis=1)

# drop any columns that contain NaN
df = df.dropna(axis=1)

# drop condition 6 (far too funky)
#df = df[df['condition'] != 6]

analytes = df.columns[2:]
df


df = df.set_index(['condition', 'bio_rep'])
df_z = (df - df.mean()) / df.std()


################################################
# Adding in cell counts ---
################################################

cell_counts = pd.read_csv('data/well_summary.csv', index_col=0)

cell_counts['bio_rep'] = cell_counts['well_id'].str.extract(r'([A-Za-z]+)')
cell_counts['condition'] = cell_counts['well_id'].str.extract(r'(\d+)').astype(int)
cell_counts = cell_counts.drop(['well_id'], axis=1)

cell_counts.head()

# MERGE

merged_df = df.reset_index().merge(
    cell_counts,
    on=['condition', 'bio_rep']
)
#df = df.set_index(['condition', 'bio_rep'])
merged_df.to_csv('data/luminex_imaging_counts_merged.csv')
