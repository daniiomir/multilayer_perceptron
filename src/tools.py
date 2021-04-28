import pickle
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class StandartScaler:
    def __init__(self):
        self.columns = []
        self.mean = {}
        self.std = {}

    def _fit(self, data: pd.DataFrame):
        for col in data.columns:
            self.mean[col] = np.mean(data[col].values)
            self.std[col] = np.std(data[col].values, ddof=1)
        self.columns = data.columns.to_list()
        self.columns.sort()

    def _scale(self, value, mean, std):
        return (value - mean) / std

    def fit_transform(self, data: pd.DataFrame):
        self._fit(data)
        return self.transform(data)

    def transform(self, data: pd.DataFrame):
        cols = data.columns.to_list()
        cols.sort()
        if cols == self.columns:
            data = data.copy()
            for col in data.columns:
                data[col] = data[col].apply(self._scale, mean=self.mean[col], std=self.std[col])
            return data
        raise Exception('Dataframe columns are not equal to previous.')


class LabelEncoder:
    def __init__(self):
        self.mapping = None

    def _fit(self, targets: np.ndarray):
        classes = np.unique(targets)
        self.mapping = {classes[i]: i for i in range(len(classes))}

    def fit_transform(self, targets: np.ndarray):
        self._fit(targets)
        return self.transform(targets)

    def transform(self, targets: np.ndarray):
        if self.mapping is not None:
            encoded = np.zeros(targets.shape, dtype=int)
            for k, v in self.mapping.items():
                encoded[targets == k] = v
            return encoded
        raise Exception('You should do fit_transform first!')

    def reverse_transform(self, reversed_targets: np.ndarray):
        if self.mapping is not None:
            targets = np.zeros(reversed_targets.shape, dtype=object)
            for k, v in self.mapping.items():
                targets[reversed_targets == v] = k
            return targets
        raise Exception('You should do fit_transform first!')


def most_correlated_features(df: pd.DataFrame, threshold: float):
    cor_matrix = df.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    return [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]


def make_corr_heatmap(dataset: pd.DataFrame):
    corr = dataset.corr()
    plt.figure(figsize=(20, 18))
    sns.heatmap(corr,
                xticklabels=corr.columns,
                yticklabels=corr.columns, annot=True)
    plt.tight_layout()
    plt.savefig('imgs/corr_plot.png')
    plt.close()


def make_count_plot(df: pd.DataFrame, df_col: pd.Series):
    sns.countplot(x=df_col, data=df)
    plt.tight_layout()
    plt.savefig('imgs/count_plot_labels.png')
    plt.close()


def parse_args_preprocess():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='data/data.csv')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    return args.__dict__


def parse_args_train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_df_path', default='tmp/train.pkl')
    parser.add_argument('--save_model_path', default='tmp/model.pkl')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    return args.__dict__


def parse_args_test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_df_path', default='tmp/test.pkl')
    parser.add_argument('--load_model_path', default='tmp/model.pkl')
    args = parser.parse_args()
    return args.__dict__


def save(obj: object, path: str):
    with open(path, 'wb') as f:
        pickle.dump(obj, f, protocol=4)


def load(path: str):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj


if __name__ == '__main__':
    pass
