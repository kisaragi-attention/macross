import multiprocessing
from pathlib import Path

import datasets
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ml.dataset import dataset_collate_function


def confusion_matrix(data_path, model, num_class):
    data_path = Path(data_path)
    model.eval()

    cm = np.zeros((num_class, num_class), dtype=np.float)

    dataset_dict = datasets.load_dataset(str(data_path.absolute()))
    dataset = dataset_dict[list(dataset_dict.keys())[0]]
    try:
        num_workers = multiprocessing.cpu_count()
    except:
        num_workers = 1
    dataloader = DataLoader(
        dataset,
        batch_size=4096,
        num_workers=num_workers,
        collate_fn=dataset_collate_function,
    )
    for batch in dataloader:
        x = batch["feature"].float().to(model.device)
        y = batch["label"].long()
        y_hat = torch.argmax(F.log_softmax(model(x), dim=1), dim=1)

        for i in range(len(y)):
            cm[y[i], y_hat[i]] += 1

    return cm


def get_precision(cm, i):
    tp = cm[i, i]
    tp_fp = cm[:, i].sum()

    return tp / tp_fp


def get_recall(cm, i):
    tp = cm[i, i]
    p = cm[i, :].sum()

    return tp / p

def get_accuracy(cm):
    tp_sum = np.trace(cm)  
    total_sum = cm.sum()  

    return tp_sum / total_sum if total_sum > 0 else 0.0 


def get_f1score(cm, i):
    precision = get_precision(cm, i)
    recall = get_recall(cm, i)
    
    return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0 

def get_macro_precision(cm):
    if not isinstance(cm, np.ndarray) or cm.size == 0 or cm.ndim !=2 or cm.shape[0] != cm.shape[1]:
        return None
    num_classes = cm.shape[0]
    macro_precision = 0.0
    for i in range(num_classes):
        precision = get_precision(cm, i)
        macro_precision += precision
    return macro_precision / num_classes

def get_macro_recall(cm):
    if not isinstance(cm, np.ndarray) or cm.size == 0 or cm.ndim !=2 or cm.shape[0] != cm.shape[1]:
        return None
    num_classes = cm.shape[0]
    macro_recall = 0.0
    for i in range(num_classes):
        recall = get_recall(cm, i)
        macro_recall += recall
    return macro_recall / num_classes

def get_macro_f1score(cm):
    if not isinstance(cm, np.ndarray) or cm.size == 0 or cm.ndim !=2 or cm.shape[0] != cm.shape[1]:
        return None
    num_classes = cm.shape[0]
    macro_f1score = 0.0
    for i in range(num_classes):
        precision = get_precision(cm, i)
        recall = get_recall(cm, i)
        f1score = get_f1score(precision, recall)
        macro_f1score += f1score
    return macro_f1score / num_classes

def get_micro_precision(cm):
    if not isinstance(cm, np.ndarray) or cm.size == 0 or cm.ndim !=2 or cm.shape[0] != cm.shape[1]:
        return None
    tp_total = np.diag(cm).sum()  
    tp_fp_total = cm.sum()        
    if tp_fp_total == 0:
        return 0.0
    return tp_total / tp_fp_total

def get_micro_recall(cm):
    if not isinstance(cm, np.ndarray) or cm.size == 0 or cm.ndim !=2 or cm.shape[0] != cm.shape[1]:
        return None
    tp_total = np.diag(cm).sum() 
    tp_fn_total = cm.sum()      
    if tp_fn_total == 0:
        return 0.0
    return tp_total / tp_fn_total

def get_micro_f1score(cm):
    micro_precision = get_micro_precision(cm)
    micro_recall = get_micro_recall(cm)
    if micro_precision is None or micro_recall is None:
        return None
    if micro_precision + micro_recall == 0:
        return 0.0
    return 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

def get_classification_report(cm, labels=None):
    rows = []
    for i in range(cm.shape[0]):
        precision = get_precision(cm, i)
        recall = get_recall(cm, i)
        accuracy = get_accuracy(cm)
        f1score = get_f1score(cm, i)
        if labels:
            label = labels[i]
        else:
            label = i

        row = {"label": label, "accuracy": accuracy, "f1score": f1score, "precision": precision, "recall": recall}
        rows.append(row)

    return pd.DataFrame(rows)
