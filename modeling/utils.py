from copy import deepcopy
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split
import torch.nn as nn


# Not using seed to ensure that as many different segmentation masks are chosen at random
def split_dataset(dataset, val_size):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_size)

    # # Copy the dataset into a new object, and set train mode for the old object
    # # NOTE: This is one "hack" to make sure val dataset doesn't have training augmentations
    # dataset_ = deepcopy(dataset)
    # dataset.set_train_mode()
    # train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset_, val_idx)

    train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    print('Divided base dataset of size %d into %d [1] and %d [2] sub-datasets!' %
          (len(dataset), len(train_dataset), len(val_dataset)))

    return train_dataset, val_dataset


def truncated_normal_(tensor, mean=0.0, std=1.0):
    size = tensor.shape
    temp = tensor.new_empty(size + (4,)).normal_()
    valid = (temp < 2) & (temp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(temp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


def init_weights(m):
    if type(m) == nn.Conv3d:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        truncated_normal_(m.bias, mean=0.0, std=0.001)
        # nn.init.normal_(m.weight, std=0.001)
        # nn.init.normal_(m.bias, std=0.001)


def init_weights_orthogonal_normal(m):
    if type(m) == nn.Conv3d:
        nn.init.orthogonal_(m.weight)
        truncated_normal_(m.bias, mean=0.0, std=0.001)


def l2_regularization(m):
    l2_reg = None
    for w in m.parameters():
        if l2_reg is None:
            l2_reg = w.norm(2)
        else:
            l2_reg = l2_reg + w.norm(2)

    return l2_reg


