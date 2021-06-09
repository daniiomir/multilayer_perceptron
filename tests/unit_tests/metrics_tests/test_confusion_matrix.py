import numpy as np
from sklearn.metrics import confusion_matrix
from src.utils.metrics import nn_confusion_matrix


def test_conf_matrix_binary1():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 1])
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()


def test_conf_matrix_binary2():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 1, 1])
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()


def test_conf_matrix_binary3():
    y_true = np.array([0, 1, 1, 1])
    y_pred = np.array([0, 1, 0, 1])
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()


def test_conf_matrix_multiclass1():
    y_true = np.random.randint(low=0, high=10, size=(100,))
    y_pred = np.random.randint(low=0, high=10, size=(100,))
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()


def test_conf_matrix_multiclass2():
    y_true = np.random.randint(low=0, high=20, size=(1000,))
    y_pred = np.random.randint(low=0, high=20, size=(1000,))
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()


def test_conf_matrix_multiclass3():
    y_true = np.random.randint(low=0, high=100, size=(10000,))
    y_pred = np.random.randint(low=0, high=100, size=(10000,))
    my_conf = nn_confusion_matrix(y_true, y_pred)
    sk_conf = confusion_matrix(y_true, y_pred)
    assert (my_conf == sk_conf).all()
