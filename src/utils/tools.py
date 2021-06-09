import pickle
import numpy as np
import pandas as pd


def most_correlated_features(df: pd.DataFrame, threshold: float) -> list:
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]


def save(obj: object, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj
