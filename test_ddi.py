import torch
import torch.nn as nn
import numpy as np
import csv
import os
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from figure import draw_whole_confusion_matrix, draw_sub_confusion_matrix, draw_sub_class_accuracy_curve

def test_model(model, test_loader, event_num, device):
    model.eval()
    pred = np.zeros((0, event_num), dtype=float)
    with torch.no_grad():
        for *modalities, _ in test_loader:
            modalities = [m.to(device) for m in modalities]
            outputs, attention_maps = model(*modalities)
            outputs = nn.Softmax(dim=1)(outputs)
            pred = np.vstack((pred, outputs.cpu().numpy()))
    return pred

def save_result(feature_name, result_type, clf_type, result, saved_path):
    with open(saved_path + feature_name + '_' + result_type + '_' + clf_type+ '.csv', "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in result:
            writer.writerow(i)
    return 0

def evaluate(pred_type, pred_score, y_test, event_num, task, saved_path):
    all_eval_type = 11
    result_all = np.zeros((all_eval_type, 1), dtype=float)
    each_eval_type = 6
    result_keep = np.zeros((each_eval_type, 1), dtype=float)
    result_eve = np.zeros((event_num, each_eval_type), dtype=float)
    y_one_hot = label_binarize(y_test, classes=np.arange(event_num))
    pred_one_hot = label_binarize(pred_type, classes=np.arange(event_num))

    save_raw_data(
        saved_path,
        f'{task}_evaluation_data',
        y_true=y_test,
        y_pred=pred_type,
        y_score=pred_score,
        event_num=np.array([event_num]) # Save scalar as array
    )

    result_all[0] = accuracy_score(y_test, pred_type)
    result_all[1] = roc_aupr_score(y_one_hot, pred_score, average='micro')
    result_all[2] = roc_aupr_score(y_one_hot, pred_score, average='macro')
    result_all[3] = roc_auc_score(y_one_hot, pred_score, average='micro')

    try:
        result_all[4] = roc_auc_score(y_one_hot, pred_score, average='macro')
    except ValueError:
        scores = []
        for i in range(y_one_hot.shape[1]):
            y_true_i = y_one_hot[:, i]
            y_pred_i = pred_score[:, i]
            if np.unique(y_true_i).size < 2:
                continue
            scores.append(roc_auc_score(y_true_i, y_pred_i))
        result_all[4] = np.mean(scores) if scores else float('nan')

    result_all[5] = f1_score(y_test, pred_type, average='micro', zero_division=0)
    result_all[6] = f1_score(y_test, pred_type, average='macro', zero_division=0)
    result_all[7] = precision_score(y_test, pred_type, average='micro', zero_division=0)
    result_all[8] = precision_score(y_test, pred_type, average='macro', zero_division=0)
    result_all[9] = recall_score(y_test, pred_type, average='micro', zero_division=0)
    result_all[10] = recall_score(y_test, pred_type, average='macro', zero_division=0)

    print("accuracy_score:", result_all[0])
    print("roc_aupr_score_micro:", result_all[1])
    print("roc_auc_score_micro:", result_all[3])
    print("f1_score_macro:", result_all[6])
    print("precision_score_macro:", result_all[8])
    print("recall_score_macro:", result_all[10])

    result_keep[0] = result_all[0]
    result_keep[1] = result_all[1]
    result_keep[2] = result_all[3]
    result_keep[3] = result_all[6]
    result_keep[4] = result_all[8]
    result_keep[5] = result_all[10]

    for i in range(event_num):
        result_eve[i, 0] = accuracy_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel())
        result_eve[i, 1] = roc_aupr_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                          average=None)
        try:
            result_eve[i, 2] = roc_auc_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                             average=None)
        except ValueError:
            result_eve[i, 2] = 0.0

        result_eve[i, 3] = f1_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                    average='binary', zero_division=0)
        result_eve[i, 4] = precision_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                           average='binary', zero_division=0)
        result_eve[i, 5] = recall_score(y_one_hot.take([i], axis=1).ravel(), pred_one_hot.take([i], axis=1).ravel(),
                                        average='binary', zero_division=0)

    draw_whole_confusion_matrix(y_test, pred_type, event_num, task, saved_path)
    draw_sub_confusion_matrix(y_test, pred_type, event_num, task, saved_path)
    draw_sub_class_accuracy_curve(y_test, pred_type, event_num, task, saved_path)

    return [result_keep, result_all, result_eve]


def self_metric_calculate(y_true, pred_type):
    y_true = y_true.ravel()
    y_pred = pred_type.ravel()
    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_pred_c = y_pred.take([0], axis=1).ravel()
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for i in range(len(y_true_c)):
        if (y_true_c[i] == 1) and (y_pred_c[i] == 1):
            TP += 1
        if (y_true_c[i] == 1) and (y_pred_c[i] == 0):
            FN += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 1):
            FP += 1
        if (y_true_c[i] == 0) and (y_pred_c[i] == 0):
            TN += 1
    print("TP=", TP, "FN=", FN, "FP=", FP, "TN=", TN)
    return (TP / (TP + FP), TP / (TP + FN))


def multiclass_precision_recall_curve(y_true, y_score):
    y_true = y_true.ravel()
    y_score = y_score.ravel()

    if y_true.ndim == 1:
        y_true = y_true.reshape((-1, 1))
    if y_score.ndim == 1:
        y_score = y_score.reshape((-1, 1))
    y_true_c = y_true.take([0], axis=1).ravel()
    y_score_c = y_score.take([0], axis=1).ravel()
    precision, recall, pr_thresholds = precision_recall_curve(y_true_c, y_score_c)

    return (precision, recall, pr_thresholds)

def roc_aupr_score(y_true, y_score, average="macro"):
    def _binary_roc_aupr_score(y_true, y_score):
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_score)
        return auc(recall, precision)

    def _average_binary_score(binary_metric, y_true, y_score, average):  # y_true= y_one_hot
        if average == "binary":
            return binary_metric(y_true, y_score)
        if average == "micro":
            y_true = y_true.ravel()
            y_score = y_score.ravel()
        if y_true.ndim == 1:
            y_true = y_true.reshape((-1, 1))
        if y_score.ndim == 1:
            y_score = y_score.reshape((-1, 1))
        n_classes = y_score.shape[1]
        score = np.zeros((n_classes,))
        for c in range(n_classes):
            y_true_c = y_true.take([c], axis=1).ravel()
            y_score_c = y_score.take([c], axis=1).ravel()
            score[c] = binary_metric(y_true_c, y_score_c)
        return np.average(score)

    return _average_binary_score(_binary_roc_aupr_score, y_true, y_score, average)

def save_raw_data(saved_path, filename, **kwargs):
    """
    Saves multiple numpy arrays into a compressed .npz file.
    Usage: save_raw_data(path, 'result_data', loss=train_loss, acc=accuracy)
    """
    file_path = os.path.join(saved_path, f"{filename}.npz")
    data_dict = {k: np.array(v) for k, v in kwargs.items()}
    np.savez_compressed(file_path, **data_dict)
    print(f"Raw data saved to: {file_path}")