import numpy as np
from src.utils.metrics import nn_accuracy_score
from sklearn.metrics import accuracy_score


def test_accuracy1():
    y_true = np.random.randint(low=0, high=2, size=(1000,))
    y_pred = np.random.randint(low=0, high=2, size=(1000,))
    my_acc = nn_accuracy_score(y_true, y_pred)
    sk_acc = accuracy_score(y_true, y_pred)
    assert my_acc == sk_acc


def test_accuracy2():
    y_true = np.random.randint(low=0, high=10, size=(1000,))
    y_pred = np.random.randint(low=0, high=10, size=(1000,))
    my_acc = nn_accuracy_score(y_true, y_pred)
    sk_acc = accuracy_score(y_true, y_pred)
    assert my_acc == sk_acc


def test_accuracy3():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_acc = nn_accuracy_score(y_true, y_pred)
    sk_acc = accuracy_score(y_true, y_pred)
    assert my_acc == sk_acc
