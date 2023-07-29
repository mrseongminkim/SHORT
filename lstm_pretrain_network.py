from ast import literal_eval

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
        self.vocab_size = VOCAB_SIZE
        self.regex_embedding_dim = REGEX_EMBEDDING_DIMENSION
        self.lstm_dim = LSTM_DIMENSION
        self.embedding_lstm = EmbeddingWithLSTM()
        self.embedding = self.embedding_lstm.embed
        self.lstm = nn.LSTM(self.regex_embedding_dim, self.lstm_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.lstm_dim * 2 * 2, 60)
        self.fc2 = nn.Linear(60, 20)
        self.fc3 = nn.Linear(20, 2)

    def forward(self, regex1, regex2):
        regex1 = self.embedding_lstm(regex1)
        regex1 = regex1.mean(1)

        regex2 = self.embedding(regex2)
        regex2, _ = self.lstm(regex2)
        regex2 = regex2.mean(1)

        concat = torch.cat((regex1, regex2), 1)
        concat = F.tanh(self.fc1(concat))
        concat = F.tanh(self.fc2(concat))
        concat = self.fc3(concat)
        return F.log_softmax(concat, dim=1)

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
            label = torch.LongTensor([1, 0])
        else:
            label = torch.LongTensor([0, 1])
        return regex1, regex2, label

epochs = 5000
batch_size = 256
lr = 0.1

train_size = 500_000
valid_size = 20_000
regex_data = RegexDataset("./annotations_file.csv")

train_data = Subset(regex_data, torch.arange(train_size))
valid_data = Subset(regex_data, torch.arange(train_size, train_size + valid_size))
train = DataLoader(train_data, batch_size=batch_size, shuffle=True)
valid = DataLoader(valid_data, batch_size=batch_size, shuffle=False)

model = RegexNNet().cuda()
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)

minimum_loss = float("inf")

print(f"Train on {train_size}, validate on {valid_size} samples.")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    count = 0
    for regex1, regex2, label in train:
        regex1 = regex1.cuda()
        regex2 = regex2.cuda()
        label = label.cuda()
        result = model(regex1, regex2)
        loss = criterion(result, label[:, 1])
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
        for regex1, regex2, label in valid:
            regex1 = regex1.cuda()
            regex2 = regex2.cuda()
            label = label.cuda()
            result = model(regex1, regex2)
            loss = criterion(result, label[:, 1])
            total_loss += loss.item()
            count += 1
        print(f"Valid - Epoch {epoch + 1}, loss: {total_loss / count}")
        if total_loss < minimum_loss:
            minimum_loss = total_loss
            torch.save(model.embedding_lstm.state_dict(), f"./{epoch}embed_lstm.pth")
