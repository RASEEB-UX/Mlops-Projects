import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def scatter_plot(data:pd.DataFrame, x:str, y:str):

    sns.set_palette("Set1")
    plt.figure(figsize=(10, 6))
    jointplot = sns.jointplot(x=x, y=y, data=data, kind='scatter', height=6, ratio=3, marginal_ticks=True)

    plt.suptitle(f"Join plot of {x} and {y}", fontsize=14)

    sns.regplot(x=x, y=y, data=data, scatter=False, ax=jointplot.ax_joint, color='r')
    # plt.tight_layout()
    plt.show()
    
    
def correlation_matrix(data:pd.DataFrame):
    
    numercial_data = data.select_dtypes(include=['int64', 'float64'])
    corr_matrix = numercial_data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix", fontsize=14)
    plt.show()


def hexa_bin(data:pd.DataFrame,x:str,y:str):
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="ticks")
    sns.jointplot(x=x, y=y, data=data, kind="hex", color="#4CB391")
    plt.suptitle(f"Hexa bin plot of {x} and {y}", fontsize=14)
    plt.show()
    return None




