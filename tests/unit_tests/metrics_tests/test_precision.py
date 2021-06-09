import numpy as np
from src.utils.metrics import nn_precision_score
from sklearn.metrics import precision_score


def test_precision1():
    y_true = np.random.randint(low=0, high=2, size=(1000,))
    y_pred = np.random.randint(low=0, high=2, size=(1000,))
    my_prec = nn_precision_score(y_true, y_pred, mode='binary')
    sk_prec = precision_score(y_true, y_pred, average='binary')
    assert my_prec == sk_prec


def test_precision2():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_prec = nn_precision_score(y_true, y_pred, mode='micro')
    sk_prec = precision_score(y_true, y_pred, average='micro')
    assert my_prec == sk_prec


def test_precision3():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_prec = nn_precision_score(y_true, y_pred, mode='micro')
    sk_prec = precision_score(y_true, y_pred, average='micro')
    assert my_prec == sk_prec


def test_precision4():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_prec = nn_precision_score(y_true, y_pred, mode='macro')
    sk_prec = precision_score(y_true, y_pred, average='macro')
    assert my_prec == sk_prec


def test_precision5():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_prec = nn_precision_score(y_true, y_pred, mode='macro')
    sk_prec = precision_score(y_true, y_pred, average='macro')
    assert my_prec == sk_prec
