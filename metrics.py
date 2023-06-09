import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def tpr_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 1) & (y_true == 1) & (is_protected == 1)) / np.sum((y_true == 1) & (is_protected == 1))


def tpr_non_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 1) & (y_true == 1) & (is_protected == 0)) / np.sum((y_true == 1) & (is_protected == 0))


def tnr_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 0) & (y_true == 0) & (is_protected == 1)) / np.sum((y_true == 0) & (is_protected == 1))


def tnr_non_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 0) & (y_true == 0) & (is_protected == 0)) / np.sum((y_true == 0) & (is_protected == 0))


def fpr_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 1) & (y_true == 0) & (is_protected == 1)) / np.sum((y_true == 0) & (is_protected == 1))


def fpr_non_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 1) & (y_true == 0) & (is_protected == 0)) / np.sum((y_true == 0) & (is_protected == 0))


def fnr_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 0) & (y_true == 1) & (is_protected == 1)) / np.sum((y_true == 1) & (is_protected == 1))


def fnr_non_protected(y_true, y_pred, is_protected):
    return np.sum((y_pred == 0) & (y_true == 1) & (is_protected == 0)) / np.sum((y_true == 1) & (is_protected == 0))


def delta_fpr(y_true, y_pred, is_protected):
    return fpr_non_protected(y_true, y_pred, is_protected) - fpr_protected(y_true, y_pred, is_protected)


def delta_fnr(y_true, y_pred, is_protected):
    return fnr_non_protected(y_true, y_pred, is_protected) - fnr_protected(y_true, y_pred, is_protected)


def equalized_odds(y_true, y_pred, is_protected):
    return abs(delta_fpr(y_true, y_pred, is_protected)) + abs(delta_fnr(y_true, y_pred, is_protected))


def calculate_metrics(y_true, y_pred, is_protected):
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Bal. Acc.': balanced_accuracy_score(y_true, y_pred),
        'Eq.Odds': equalized_odds(y_true, y_pred, is_protected),
        'TPR Prot.': tpr_protected(y_true, y_pred, is_protected),
        'TPR Non-Prot.': tpr_non_protected(y_true, y_pred, is_protected),
        'TNR Prot.': tnr_protected(y_true, y_pred, is_protected),
        'TNR Non-Prot.': tnr_non_protected(y_true, y_pred, is_protected)
    }
