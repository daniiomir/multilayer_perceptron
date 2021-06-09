import numpy as np
from typing import Union, Tuple


def nn_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, mode: str = 'matrix') -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    unique_classes = np.unique(y_true)
    len_uniq = len(unique_classes)
    if y_true.shape != y_pred.shape:
        raise Exception('y_true and y_pred should be equal in shape')
    if len_uniq <= 1:
        raise Exception('in y_true should be more than 1 class')
    conf = np.zeros(shape=(len_uniq, len_uniq), dtype=np.int32)
    for index, item in enumerate(y_true):
        conf[item, y_pred[index]] += 1
    if mode == 'all':
        conf_dict = {}
        for label in range(len_uniq):
            tp = conf[label, label]
            fp = np.sum(conf[label, :]) - conf[label, label]
            fn = np.sum(conf[:, label]) - conf[label, label]
            tn = np.sum(conf) - (fp + fn + tp)
            conf_dict[label] = (tp, tn, fp, fn)
        return conf, conf_dict
    return conf


def nn_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(y_true == y_pred)


def nn_precision_score(y_true: np.ndarray, y_pred: np.ndarray, mode: str = 'binary') -> float:
    conf, conf_dict = nn_confusion_matrix(y_true, y_pred, mode='all')
    if mode == 'binary':
        tp, tn, fp, fn = conf_dict[0]
        return tp / (tp + fp)
    elif mode == 'micro':
        all_tp = 0
        all_fp = 0
        for label, (tp, tn, fp, fn) in conf_dict.items():
            all_tp += tp
            all_fp += fp
        return all_tp / (all_tp + all_fp)
    elif mode == 'macro':
        class_metrics = [tp / (tp + fp) for label, (tp, tn, fp, fn) in conf_dict.items()]
        return np.mean(class_metrics)
    else:
        raise NotImplementedError


def nn_recall_score(y_true: np.ndarray, y_pred: np.ndarray, mode: str = 'binary') -> float:
    conf, conf_dict = nn_confusion_matrix(y_true, y_pred, mode='all')
    if mode == 'binary':
        tp, tn, fp, fn = conf_dict[0]
        return tp / (tp + fn)
    elif mode == 'micro':
        all_tp = 0
        all_fn = 0
        for label, (tp, tn, fp, fn) in conf_dict.items():
            all_tp += tp
            all_fn += fn
        return all_tp / (all_tp + all_fn)
    elif mode == 'macro':
        class_metrics = [tp / (tp + fn) for label, (tp, tn, fp, fn) in conf_dict.items()]
        return np.mean(class_metrics)
    else:
        raise NotImplementedError


def nn_f1_score(y_true: np.ndarray, y_pred: np.ndarray, mode: str = 'binary') -> float:
    prec = nn_precision_score(y_true, y_pred, mode)
    rec = nn_recall_score(y_true, y_pred, mode)
    f1 = (2 * prec * rec) / (prec + rec)
    return f1


if __name__ == '__main__':
    y_true = np.random.randint(low=0, high=2, size=(40,))
    y_pred = np.random.randint(low=0, high=2, size=(40,))
    conf, conf_dict = nn_confusion_matrix(y_true, y_pred, mode='all')
    print(conf)
    print(conf_dict)
