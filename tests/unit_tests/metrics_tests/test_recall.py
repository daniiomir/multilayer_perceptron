import numpy as np
from src.utils.metrics import nn_recall_score
from sklearn.metrics import recall_score


def test_recall1():
    y_true = np.random.randint(low=0, high=2, size=(1000,))
    y_pred = np.random.randint(low=0, high=2, size=(1000,))
    my_rec = nn_recall_score(y_true, y_pred, mode='binary')
    sk_rec = recall_score(y_true, y_pred, average='binary')
    assert my_rec == sk_rec


def test_recall2():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_rec = nn_recall_score(y_true, y_pred, mode='micro')
    sk_rec = recall_score(y_true, y_pred, average='micro')
    assert my_rec == sk_rec


def test_recall3():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_rec = nn_recall_score(y_true, y_pred, mode='micro')
    sk_rec = recall_score(y_true, y_pred, average='micro')
    assert my_rec == sk_rec


def test_recall4():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_rec = nn_recall_score(y_true, y_pred, mode='macro')
    sk_rec = recall_score(y_true, y_pred, average='macro')
    assert my_rec == sk_rec


def test_recall5():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_rec = nn_recall_score(y_true, y_pred, mode='macro')
    sk_rec = recall_score(y_true, y_pred, average='macro')
    assert my_rec == sk_rec
