import numpy as np
import pandas as pd


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


if __name__ == '__main__':
    pass
