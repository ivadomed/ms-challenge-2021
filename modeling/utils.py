from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def split_dataset(dataset, val_size, seed):
    train_idx, val_idx = train_test_split(list(range(len(dataset))),
                                          test_size=val_size,
                                          random_state=seed)

    train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    print('Divided base dataset of size %d into %d [1] and %d [2] sub-datasets!' %
          (len(dataset), len(train_dataset), len(val_dataset)))

    return train_dataset, val_dataset