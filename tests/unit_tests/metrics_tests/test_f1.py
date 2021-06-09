import numpy as np
from src.utils.metrics import nn_f1_score
from sklearn.metrics import f1_score


def test_f1_score1():
    y_true = np.random.randint(low=0, high=2, size=(1000,))
    y_pred = np.random.randint(low=0, high=2, size=(1000,))
    my_f1 = nn_f1_score(y_true, y_pred, mode='binary')
    sk_f1 = f1_score(y_true, y_pred, average='binary')
    assert my_f1 == sk_f1


def test_f1_score2():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_f1 = nn_f1_score(y_true, y_pred, mode='micro')
    sk_f1 = f1_score(y_true, y_pred, average='micro')
    assert my_f1 == sk_f1


def test_f1_score3():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_f1 = nn_f1_score(y_true, y_pred, mode='micro')
    sk_f1 = f1_score(y_true, y_pred, average='micro')
    assert my_f1 == sk_f1


def test_f1_score4():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_f1 = nn_f1_score(y_true, y_pred, mode='macro')
    sk_f1 = f1_score(y_true, y_pred, average='macro')
    assert my_f1 == sk_f1


def test_f1_score5():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_f1 = nn_f1_score(y_true, y_pred, mode='macro')
    sk_f1 = f1_score(y_true, y_pred, average='macro')
    assert my_f1 == sk_f1
