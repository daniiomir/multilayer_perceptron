import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def make_corr_heatmap(dataset: pd.DataFrame) -> None:
    corr = dataset.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True)
    plt.tight_layout()
    plt.savefig('../imgs/corr_plot.png')
    plt.close()


def make_count_plot(df: pd.DataFrame, df_col: pd.Series) -> None:
    sns.countplot(x=df_col, data=df)
    plt.tight_layout()
    plt.savefig('../imgs/count_plot_labels.png')
    plt.close()


def make_learning_curves(train_list: list, val_list: list, name: str):
    plt.plot(np.arange(len(train_list)), train_list, color='blue', label='train')
    plt.plot(np.arange(len(val_list)), val_list, color='red', label='val')
    plt.legend(loc='best')
    plt.title(f'Development of {name.capitalize()} during training')
    plt.xlabel('Number of iterations')
    plt.ylabel(name.capitalize())
    plt.savefig(f'../imgs/{name}.png')
    plt.close()
