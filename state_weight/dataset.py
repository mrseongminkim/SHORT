import os
import pickle
from ast import literal_eval

import torch
from torch_geometric.data import Dataset

import pandas as pd

class StateWeightDataset(Dataset):
    def __init__(self, annotations_file, fa_dir, fa_transform=None):
        super().__init__(annotations_file)
        self.fa_labels = pd.read_csv(annotations_file, names=["fa", "label"], converters={"label": literal_eval})
        self.fa_dir = fa_dir
        self.fa_transform = fa_transform

    def len(self):
        return len(self.fa_labels)
    
    def get(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        fa_path = os.path.join(self.fa_dir, self.fa_labels.iloc[idx, 0])
        with open(fa_path, "rb") as fp:
            fa = pickle.load(fp)
        label = torch.FloatTensor(self.fa_labels.iloc[idx, 1])
        if self.fa_transform:
            fa = self.fa_transform(fa)
        return fa, label
