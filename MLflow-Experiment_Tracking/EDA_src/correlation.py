import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def correlation_matrix(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 8))

    numerical_df = df.select_dtypes(include=[np.number])
    
    corr_matrix = numerical_df.corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    
    plt.show()