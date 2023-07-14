import os
import pickle

import torch
import pandas as pd
from torch.utils.data import Dataset, Subset, DataLoader

class MembershipTestDataset(Dataset):
    def __init__(self, annotations_file, fa_dir, fa_transform=None, word_transform=None, target_trasform=None):
        self.fa_labels = pd.read_csv(annotations_file, names=["fa", "word", "label"])
        self.fa_dir = fa_dir
        self.fa_transform = fa_transform
        self.word_transform = word_transform
        self.target_transform = target_trasform
    
    def __len__(self):
        return len(self.fa_labels)
    
    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        fa_path = os.path.join(self.fa_dir, self.fa_labels.iloc[idx, 0])
        with open(fa_path, "rb") as fp:
            fa = pickle.load(fp)
        word = self.fa_labels.iloc[idx, 1]
        label = self.fa_labels.iloc[idx, 2]
        if self.fa_transform:
            fa = self.fa_transform(fa)
        if self.word_transform:
            word = self.word_transform(word)
        if self.target_transform:
            label = self.target_transform(label)
        return fa, word, label

def test_transform(gfa):
    lst = [int(x) for x in gfa.States]
    lst = lst + [0 for i in range(10 - len(lst))]
    return torch.Tensor(lst)

TRAINING_DATA_SIZE = 80_000
membership_test_data = MembershipTestDataset("./membership_test_data/annotations_file.csv", "./membership_test_data/", fa_transform=test_transform)

train_data = Subset(membership_test_data, torch.arange(TRAINING_DATA_SIZE))
test_data = Subset(membership_test_data, torch.arange(TRAINING_DATA_SIZE, len(membership_test_data)))

train = DataLoader(train_data, batch_size=6, shuffle=True)
test = DataLoader(test_data, batch_size=1, shuffle=False)

for fa, word, label in train:
    print(fa, word, label)
    exit()

#TypeError: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found <class 'FAdo.conversions.GFA'>
#need to transform FAs
