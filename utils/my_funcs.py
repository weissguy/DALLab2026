import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def plot_cytokine(df: pd.DataFrame, cytokine: str):
    '''
    Plots cytokine traces

    Parameters:
    df (pd.DataFrame): df with observations as rows and cytokines as columns
    cytokine (str): cytokine to plot

    Returns:
    None
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(
        data=df,
        x='condition',
        y=cytokine,
        ax=ax
    )
    ax.set_title(f"{cytokine} Concentration by Condition")
    ax.set_xlabel("Condition")
    ax.set_ylabel("Concentration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()