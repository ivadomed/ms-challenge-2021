import random

from sklearn.model_selection import train_test_split
import torch
from typing import TypeVar, Sequence
from torch.utils.data import Dataset


T_co = TypeVar('T_co', covariant=True)
T = TypeVar('T')


class BalancedSubset(Dataset[T_co]):
    """
    Subset of a dataset at specified indices. Adapted from torch.utils.data.Subset.

    NOTE: Don't forget to call BalancedSubset().prepare_for_new_epoch() at the beginning of
    each epoch.

    :param (datasets.MSSeg2Dataset) dataset: The entire dataset. See modeling/datasets.py
    :param (list) indices: Sequence of indices to get items from `dataset`
    :param (bool) train: True for training sets and False for other sets (e.g. validation)
    :param (str) balance_strategy: Balancing strategy to-be used in training. This is important
           because lesion segmentation datasets we work on are highly imbalanced. The possible
           values and explanations for this argument are given below:
            1) 'none': does nothing, leaves the train set as is.
            2) 'naive_duplication': duplicates each positive index as many times as required to get
                                    a balanced set of 0.5 positive / 0.5 negative subvolumes.
            3) 'cheap_duplication': duplicates each positive index [self.k = 10] times to get
                                    somewhat of a more balanced set.
            3) 'naive_removal': removes as many random negative indices as required to get a
                                balanced set of 0.5 positive / 0.5 negative subvolumes.
            4) TODO: 'curriculum_learning': Dynamically shift the distribution with:
                                            0.9+ / 0.1-, ..., 0.5+ / 0.5-, ..., 0.1+ / 0.9-
    """
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset, indices, train=False, balance_strategy='none') -> None:
        self.dataset = dataset
        self.indices = indices
        self.train = train
        self.balance_strategy = balance_strategy

        if not self.train and balance_strategy != 'none':
            raise ValueError('The balance_strategy must be "none" for non-training datasets!')

        if self.train:
            all_positive_indices = dataset.positive_indices
            self.train_positive_indices, self.train_negative_indices = [], []

            # Get positive training indices and new train inbalance factor
            for i in self.indices:
                if i in all_positive_indices:
                    self.train_positive_indices.append(i)
                else:
                    self.train_negative_indices.append(i)

            train_inbalance_factor = len(self.train_negative_indices) // len(self.train_positive_indices)
            print('Factor of original training set inbalance is %d' % train_inbalance_factor)

            if self.balance_strategy == 'naive_duplication':
                new_indices = []
                for i in self.train_positive_indices:
                    new_indices.extend([i] * (train_inbalance_factor - 1))

                self.indices.extend(new_indices)

            elif self.balance_strategy == 'cheap_duplication':
                self.k = 10
                new_indices = []
                for i in self.train_positive_indices:
                    new_indices.extend([i] * self.k)

                self.indices.extend(new_indices)

            elif self.balance_strategy == 'naive_removal':
                # Get a random sample with as many negatives as positives to begin with
                self.k = 1
                self.current_train_negative_indices = random.sample(self.train_negative_indices, len(self.train_positive_indices) * self.k)
                self.indices = self.train_positive_indices * self.k + self.current_train_negative_indices

            elif self.balance_strategy != 'none':
                raise ValueError('The balance_strategy=%s is not recognized!' % balance_strategy)

            # Shuffle the indices initially for all balancing strategies
            random.shuffle(self.indices)

    def prepare_for_new_epoch(self):
        """Prepares the training set in the beginning of each epoch."""
        # NOTE: Nothing to do for 'naive_duplication', 'cheap_duplication', and 'none' cases!
        if self.train:
            if self.balance_strategy == 'naive_duplication' or self.balance_strategy == 'cheap_duplication':
                pass
            elif self.balance_strategy == 'naive_removal':
                # Update the random sample with k = 1, keeping the ratio intact!
                self.current_train_negative_indices = random.sample(self.train_negative_indices, len(self.train_positive_indices) * self.k)
                self.indices = self.train_positive_indices * self.k + self.current_train_negative_indices
            elif self.balance_strategy == 'none':
                pass

            # Shuffle the indices before each epoch for all balancing strategies
            random.shuffle(self.indices)
        else:
            raise ValueError('Called in non-training mode! This might bias the results you obtain!')

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def split_dataset(dataset, val_size, balance_strategy, seed):
    """
    Splits dataset into train and validation sets.

    :param (datasets.MSSeg2Dataset) dataset: The entire dataset. See modeling/datasets.py
    :param (float) val_size: Fraction of data to-be-used for validation, has to be bw. 0. and 1.
    :param (str) balance_strategy: Balancing strategy in train set. See `BalancedSubset()` for more details
    :param (int) seed: Seed for reproducibility (e.g. we want the same train-val split with same seed)
    """
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_size, random_state=seed)

    train_dataset = BalancedSubset(dataset, train_idx, train=True, balance_strategy=balance_strategy)
    val_dataset = BalancedSubset(dataset, val_idx, train=False)

    print('Divided base dataset of size %d into %d [1] and %d [2] sub-datasets!' % (len(dataset), len(train_dataset), len(val_dataset)))

    return train_dataset, val_dataset


def binary_accuracy(y_pred, y_true):
    """Calculates binary accuracy per batch"""
    y_pred = y_pred.view(-1, 2)
    y_true = y_true.view(-1)

    y_pred_max = torch.argmax(y_pred, dim=-1)
    correct_pred = (y_pred_max == y_true).float()

    acc = (correct_pred.sum() + 1e-5) / (len(correct_pred) + 1e-5)
    pos_acc = (correct_pred[y_true == 1].sum() + 1e-5) / (len(correct_pred[y_true == 1]) + 1e-5)
    neg_acc = (correct_pred[y_true == 0].sum() + 1e-5) / (len(correct_pred[y_true == 0]) + 1e-5)

    return acc, pos_acc, neg_acc


def set_seeds(seed):
    """Sets random seeds."""
    # TODO: We could do the same with numpy, torch, etc. as well.
    random.seed(seed)
