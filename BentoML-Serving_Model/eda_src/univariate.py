
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

def univariate_analysis(data: pd.DataFrame , col :str):
            
    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(1, 3, 1)
    sns.histplot(data[col], color='blue', kde=True)
    ax1.axvline(x=data[col].mean(), color='g', linestyle='--', linewidth=3)
    ax1.text(data[col].mean(), data[col].value_counts().max() * 0.9, "Mean", 
                horizontalalignment='left', size=20, color='black', weight='semibold')
    ax1.set_title(f'{col} Histogram', fontsize=20)
    
    ax2 = plt.subplot(1, 3, 2)
    sns.boxplot(x=data[col], color='blue')
    ax2.set_title(f'{col} Boxplot', fontsize=20)

    ax3 = plt.subplot(1, 3, 3)
    stats.probplot(data[col], dist=stats.norm, plot=ax3)
    ax3.set_title(f'{col} Q-Q plot', fontsize=20)
    sns.despine()

    mean = data[col].mean()
    std = data[col].std()
    skew = data[col].skew()
    print(f'{col} : mean: {mean:.4f}, std: {std:.4f}, skew: {skew:.4f}')