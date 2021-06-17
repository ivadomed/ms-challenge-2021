from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Subset


def split_dataset(dataset, val_size, seed):
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                          test_size=val_size,
                                          random_state=seed)

    train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    print('Divided base dataset of size %d into %d [1] and %d [2] sub-datasets!' %
          (len(dataset), len(train_dataset), len(val_dataset)))

    return train_dataset, val_dataset


def binary_accuracy(y_pred, y_true):
    """Function to calculate binary accuracy per batch"""
    y_pred = y_pred.view(-1, 2)
    y_true = y_true.view(-1)

    y_pred_max = torch.argmax(y_pred, dim=-1)
    correct_pred = (y_pred_max == y_true).float()

    acc = (correct_pred.sum() + 1e-5) / (len(correct_pred) + 1e-5)
    pos_acc = (correct_pred[y_true == 1].sum() + 1e-5) / (len(correct_pred[y_true == 1]) + 1e-5)
    neg_acc = (correct_pred[y_true == 0].sum() + 1e-5) / (len(correct_pred[y_true == 0]) + 1e-5)

    return acc, pos_acc, neg_acc
