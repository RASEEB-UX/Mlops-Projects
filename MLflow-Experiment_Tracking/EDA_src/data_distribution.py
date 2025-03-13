import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns




def plot_histograms(df, numerical_columns=None, figsize=(15, 10)):
  
    if numerical_columns is None:
        numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        numerical_columns = [col for col in numerical_columns 
                             if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not numerical_columns:
        print("No numerical columns found for plotting.")
        return

    n = len(numerical_columns)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    for idx, col in enumerate(numerical_columns):
        ax = axes_flat[idx]
        sns.kdeplot(data=df[col].dropna(), ax=ax, fill=True, color='skyblue')
        ax.set_title(f"KDE Plot of {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    
    for idx in range(n, len(axes_flat)):
        axes_flat[idx].axis('off')

    plt.tight_layout()
    plt.show()
    