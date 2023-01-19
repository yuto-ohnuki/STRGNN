import os, sys, glob
import numpy as np

from sklearn.metrics import roc_curve, precision_recall_curve, auc, accuracy_score
from sklearn.metrics import confusion_matrix


def calc_cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def AUPRC(prob, y):
    precision, recall, thresh = precision_recall_curve(y, prob)
    auprc = auc(recall, precision)
    return auprc


def AUROC(prob, y):
    fpr, tpr, thresh = roc_curve(y, prob)
    auroc = auc(fpr, tpr)
    return auroc


def ACC(prob, y):
    pred = (prob.flatten() >= 0.5).astype(int)
    acc = accuracy_score(pred, y)
    return acc


def calc_metrics(prob, y):
    auprc = AUPRC(prob, y)
    auroc = AUROC(prob, y)
    acc = ACC(prob, y)
    ret = {"AUROC": auroc, "AUPRC": auprc, "ACC": acc}
    return ret


def evaluation(pred, target):
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    metrics = calc_metrics(pred, target)
    return metrics


def calc_average_and_std(value):
    ret = np.array([max(v) for v in value])
    ave = np.mean(ret)
    std = np.std(ret)
    return ave, std


def show_max_values(auroc, auprc, acc):
    print("AUROC: {}".format(max(auroc)))
    print("AUPRC: {}".format(max(auprc)))
    print("ACC:   {}".format(max(acc)))
