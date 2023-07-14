import os
import pickle

import torch
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader

class EquivalenceTestDataset(Dataset):
    def __init__(self, annotations_file, fa_dir, fa_transform=None, target_trasform=None):
        self.fa_labels = pd.read_csv(annotations_file, names=["fa_1", "fa_2", "label"])
        self.fa_dir = fa_dir
        self.fa_transform = fa_transform
        self.target_transform = target_trasform
    
    def __len__(self):
        return len(self.fa_labels)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        fa_path_1 = os.path.join(self.fa_dir, self.fa_labels.iloc[idx, 0])
        with open(fa_path_1, "rb") as fp:
            fa_1 = pickle.load(fp)
        fa_path_2 = os.path.join(self.fa_dir, self.fa_labels.iloc[idx, 1])
        with open(fa_path_2, "rb") as fp:
            fa_2 = pickle.load(fp)
        label = self.fa_labels.iloc[idx, 2]
        if self.fa_transform:
            fa_1 = self.fa_transform(fa_1)
            fa_2 = self.fa_transform(fa_2)
        if self.target_transform:
            label = self.target_transform(label)
        return fa_1, fa_2, label

def test_transform(gfa):
    lst = [int(x) for x in gfa.States]
    return torch.Tensor(lst)

TRAINING_DATA_SIZE = 80_000
equivalence_test_data = EquivalenceTestDataset("./equivalence_test_data/annotations_file.csv", "./equivalence_test_data/", fa_transform=test_transform)

train_data = Subset(equivalence_test_data, torch.arange(TRAINING_DATA_SIZE))
test_data = Subset(equivalence_test_data, torch.arange(TRAINING_DATA_SIZE, len(equivalence_test_data)))

train = DataLoader(train_data, batch_size=6, shuffle=False)
test = DataLoader(test_data, batch_size=6, shuffle=False)

for fa_1, fa_2, label in train:
    print(fa_1, fa_2, label)
    exit()
