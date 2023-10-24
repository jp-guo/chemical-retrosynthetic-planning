import pickle
import numpy as np
import gzip

import torch
from torch.utils.data import Dataset, RandomSampler


class DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        fp, cost = self.dataset[item]
        return torch.tensor(fp), torch.tensor(cost)


def load_mol_eval_single_dataset(path: str) -> DatasetWrapper:
    file = gzip.open(path, "rb")
    dataset = pickle.load(file, encoding='latin1')
    packed_fp = dataset['packed_fp']
    fp = np.unpackbits(packed_fp, axis=1)
    cost = dataset['values']
    file.close()
    return DatasetWrapper(dataset=list(zip(fp, cost)))


def load_mol_eval_multi_dataset(single_dataset: DatasetWrapper) -> DatasetWrapper:
    fp_concats = []
    cost_sums = []
    for _ in range(10000):
        indices = RandomSampler(range(len(single_dataset)), replacement=False, num_samples=3)
        indices = [i for i in indices]
        fp_concat = torch.zeros(0)
        cost_sum = 0
        for i in indices:
            fp, cost = single_dataset[i]
            fp_concat = torch.cat((fp_concat, fp.float()))
            cost_sum += cost

        fp_concats.append(fp_concat)
        cost_sums.append(cost_sum)
    return DatasetWrapper(dataset=list(zip(fp_concats, cost_sums)))
