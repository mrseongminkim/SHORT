from tqdm import tqdm
from ast import literal_eval
from pickle import load, dump

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, Subset, DataLoader

from config import *

class EmbeddingWithLSTM(nn.Module):
    def __init__(self):
        super(EmbeddingWithLSTM, self).__init__()
        self.vocab_size = VOCAB_SIZE
        self.regex_embedding_dim = REGEX_EMBEDDING_DIMENSION
        self.lstm_dim = LSTM_DIMENSION
        self.embed = nn.Embedding(self.vocab_size, self.regex_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.regex_embedding_dim, self.lstm_dim, batch_first=True, bidirectional=True)

    def forward(self, regex):
        regex = self.embed(regex)
        regex, _ = self.lstm(regex)
        return regex

class RegexNNet(nn.Module):
    def __init__(self):
        super(RegexNNet, self).__init__()
        self.lstm_dim = LSTM_DIMENSION
        self.embedding_lstm = EmbeddingWithLSTM()
        self.embedding_lstm.load_state_dict(torch.load("0.191.pth"))

    def sub_forward(self, x):
        x = self.embedding_lstm(x)
        x = x[:, -1]
        return x

    def forward(self, x1, x2):
        x1 = self.sub_forward(x1)
        x2 = self.sub_forward(x2)
        return x1, x2

class RegexDataset(Dataset):
    def __init__(self, annotations_file):
        self.data = pd.read_csv(annotations_file, names=["regex1", "regex2", "label"], converters={"regex1": literal_eval, "regex2": literal_eval, "label": int})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        regex1 = torch.LongTensor(self.data.iloc[idx, 0])
        regex2 = torch.LongTensor(self.data.iloc[idx, 1])
        label = self.data.iloc[idx, 2]
        if label == 0:
            label = -1
        return regex1, regex2, label

torch.autograd.set_detect_anomaly(True)

train_size = 1_300_000
valid_size = 50_000
#regex_data = RegexDataset("./annotations_file.csv")
#with open("./annotations_file.pkl", "wb") as f:
#    dump(regex_data, f)
with open("./annotations_file.pkl", "rb") as f:
    regex_data = load(f)

epochs = 999999999999999999999999999999999
batch_size = 64
#0.0005로 75에폭까지 돌리고
#0.00001로 다시 돌림
lr = 0.00000001

train_data = Subset(regex_data, torch.arange(train_size))
valid_data = Subset(regex_data, torch.arange(train_size, train_size + valid_size))
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

model = RegexNNet().cuda()
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=1)

minimum_loss = float("inf")
print(f"Train on {train_size}, validate on {valid_size} samples.")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    count = 0
    for regex1, regex2, label in tqdm(train):
        regex1 = regex1.cuda()
        regex2 = regex2.cuda()
        label = label.cuda()
        x1, x2 = model(regex1, regex2)
        loss = criterion(input1=x1, input2=x2, target=label)
        total_loss += loss.item()
        count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train - Epoch {epoch + 1}, loss: {total_loss / count}")
    with torch.no_grad():
        model.eval()
        total_loss = 0
        count = 0
        for regex1, regex2, label in tqdm(valid):
            regex1 = regex1.cuda()
            regex2 = regex2.cuda()
            label = label.cuda()
            x1, x2 = model(regex1, regex2)
            loss = criterion(x1, x2, label)
            total_loss += loss.item()
            count += 1
        print(f"Valid - Epoch {epoch + 1}, loss: {total_loss / count}")
        if total_loss < minimum_loss:
            minimum_loss = total_loss
            torch.save(model.embedding_lstm.state_dict(), f"./{epoch + 1}embed_lstm.pth")
        #scheduler.step(total_loss / count)
