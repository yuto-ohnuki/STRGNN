import os, sys, glob
import numpy as np
import pandas as pd

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


def show_results(train_record, valid_record, best_record, conf):
    line = "#" * 40
    print(line)
    print("Results:")
    
    ret_indexes = []
    ret_train_aurocs, ret_train_auprcs, ret_train_accs = [], [], []
    ret_valid_aurocs, ret_valid_auprcs, ret_valid_accs = [], [], []
    
    # Training/Validation records
    for cv in range(conf.cv):
        cv_train_record = np.array(train_record[cv]).T
        cv_valid_record = np.array(valid_record[cv]).T
        best_index = np.argmax(cv_valid_record[1])
        ret_indexes.append(best_index)
        
        ret_train_aurocs.append(cv_train_record[0][best_index])
        ret_train_auprcs.append(cv_train_record[1][best_index])
        ret_train_accs.append(cv_train_record[2][best_index])
        ret_valid_aurocs.append(cv_valid_record[0][best_index])
        ret_valid_auprcs.append(cv_valid_record[1][best_index])
        ret_valid_accs.append(cv_valid_record[2][best_index])
    
    # Output as dataframe
    ret = {
        "Train: AUROC": np.array(ret_train_aurocs),
        "Train: AUPRC": np.array(ret_train_auprcs),
        "Train: ACC": np.array(ret_train_accs),
        "Valid: AUROC": np.array(ret_valid_aurocs),
        "Valid: AUPRC": np.array(ret_valid_auprcs),
        "Valid: ACC": np.array(ret_valid_accs)
    }
    
    ret = pd.DataFrame(ret)
    ret.index.name = "CV"
    display(ret)
    
    # Test performance
    best_cv = np.argmax(ret_valid_auprcs)
    result = best_record[best_cv][-1]
    
    print("\tTest:")
    perf = {
        "Test: AUROC": np.array(result["AUROC"]),
        "Test: AUPRC": np.array(result['AUPRC']),
        "Test: ACC": np.array(result["ACC"])
    }
    perf = pd.DataFrame(perf, index=[0])
    display(perf)