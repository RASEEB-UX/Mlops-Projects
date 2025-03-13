import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math



def subplot_boxplot(df: pd.DataFrame, figsize=(12, 8)):
    """
    Generates a subplot of box plots for each numerical column in the given DataFrame.

    """
    numeric_df = df.select_dtypes(include=['number'])
    numeric_columns = numeric_df.columns
    n = len(numeric_columns)
    
    if n == 0:
        print("No numerical columns found.")
        return

    ncols = min(3, n)
    nrows = math.ceil(n / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    
    axes_flat = axes.flatten()
    
    for i, col in enumerate(numeric_columns):
        sns.boxplot(data=df, y=col, ax=axes_flat[i])
        axes_flat[i].set_title(f"Box Plot of {col}")
    
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
    plt.show()