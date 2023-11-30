import os
import pickle
from ast import literal_eval
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset

from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from alpha_zero.state_elimination.gatv3 import GATv3Conv
from torch_geometric.nn.pool import global_mean_pool

from alpha_zero.utils import *

from config import *

game = StateEliminationGame()
gfa_transform = game.gfa_to_tensor

class MembershipTestDataset(Dataset):
    def __init__(self, annotations_file, fa_dir, fa_transform=None, word_transform=None, target_trasform=None):
        super().__init__(annotations_file)
        self.fa_labels = pd.read_csv(annotations_file, names=["fa", "word", "label"], converters={"word": literal_eval, "label": int})
        self.fa_dir = fa_dir
        self.fa_transform = fa_transform
        self.word_transform = word_transform
        self.target_transform = target_trasform
    
    def len(self):
        return len(self.fa_labels)
    
    def get(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()
        fa_path = os.path.join(self.fa_dir, self.fa_labels.iloc[idx, 0])
        with open(fa_path, "rb") as fp:
            fa = pickle.load(fp)
        word = torch.LongTensor(self.fa_labels.iloc[idx, 1])
        label = int(self.fa_labels.iloc[idx, 2])
        if self.fa_transform:
            fa = self.fa_transform(fa)
        if self.word_transform:
            word = self.word_transform(word)
        if self.target_transform:
            label = self.target_transform(label)
        return fa, word, label

TRAINING_DATA_SIZE = 100_000
membership_test_data = MembershipTestDataset("./membership_test_data/annotations_file.csv", "./membership_test_data/", fa_transform=gfa_transform)

train_data = membership_test_data[:TRAINING_DATA_SIZE]
test_data = membership_test_data[TRAINING_DATA_SIZE:]

train = DataLoader(train_data, batch_size=32, shuffle=True)
test = DataLoader(test_data, batch_size=32, shuffle=False)

class wordNet(nn.Module):
    def __init__(self):
        super(wordNet, self).__init__()
        self.vocab_size = 6
        self.word_embedding_dim = 2
        self.lstm_dim = 4
        self.embed = nn.Embedding(self.vocab_size, self.word_embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(self.word_embedding_dim, self.lstm_dim, batch_first=True)
    
    def forward(self, data):
        data = self.embed(data)
        data = self.lstm(data)[0][:, -1]
        return data

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

class LeftRightGNN(nn.Module):
    def __init__(self, freeze_lstm=True):
        super(LeftRightGNN, self).__init__()
        self.action_size = game.getActionSize()
        self.state_number_dim = MAX_STATES + 3
        self.lstm_dim = LSTM_DIMENSION
        self.embedding_with_lstm = EmbeddingWithLSTM()
        self.embedding_with_lstm.load_state_dict(torch.load("./alpha_zero/state_elimination/0.191.pth"))
        if freeze_lstm:
            for param in self.embedding_with_lstm.parameters():
                param.requires_grad = False
        assert NUMBER_OF_CHANNELS % NUMBER_OF_HEADS == 0
        self.conv1 = GATv3Conv(self.state_number_dim * 3 + self.lstm_dim * 2 * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv2 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv3 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv4 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv5 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)

        self.right_conv1 = GATv3Conv(self.state_number_dim * 3 + self.lstm_dim * 2 * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv2 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv3 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv4 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv5 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)

    def forward(self, left_batch, right_batch):
        regex = left_batch.edge_attr[:, :MAX_LEN]
        regex = self.embedding_with_lstm(regex)
        regex = regex[:, -1]
        #Last Time Step Pooling <-> Mean Pooling
        source_state_numbers = left_batch.edge_attr[:, MAX_LEN:MAX_LEN + MAX_STATES + 3]
        target_state_numbers = left_batch.edge_attr[:, MAX_LEN + MAX_STATES + 3:]

        left_batch.edge_attr = regex
        right_batch.edge_attr = regex

        source_states = left_batch.edge_index[0]
        target_states = left_batch.edge_index[1]

        left_out_transitions = global_mean_pool(torch.cat((target_state_numbers, regex), dim=-1), source_states, left_batch.x.size()[0])
        left_in_transitions = global_mean_pool(torch.cat((source_state_numbers, regex), dim=-1), target_states, left_batch.x.size()[0])

        left_batch.x = torch.cat((left_batch.x, left_in_transitions, left_out_transitions), dim=-1)
        right_batch.x = torch.cat((right_batch.x, left_out_transitions, left_in_transitions), dim=-1)

        left = F.elu(self.conv1(x=left_batch.x, edge_index=left_batch.edge_index, edge_attr=left_batch.edge_attr))
        left = F.elu(self.conv2(x=left, edge_index=left_batch.edge_index, edge_attr=left_batch.edge_attr))
        left = F.elu(self.conv3(x=left, edge_index=left_batch.edge_index, edge_attr=left_batch.edge_attr))
        left = F.elu(self.conv4(x=left, edge_index=left_batch.edge_index, edge_attr=left_batch.edge_attr))
        left = F.elu(self.conv5(x=left, edge_index=left_batch.edge_index, edge_attr=left_batch.edge_attr))

        right = F.elu(self.conv1(x=right_batch.x, edge_index=right_batch.edge_index, edge_attr=right_batch.edge_attr))
        right = F.elu(self.conv2(x=right, edge_index=right_batch.edge_index, edge_attr=right_batch.edge_attr))
        right = F.elu(self.conv3(x=right, edge_index=right_batch.edge_index, edge_attr=right_batch.edge_attr))
        right = F.elu(self.conv4(x=right, edge_index=right_batch.edge_index, edge_attr=right_batch.edge_attr))
        right = F.elu(self.conv5(x=right, edge_index=right_batch.edge_index, edge_attr=right_batch.edge_attr))

        data = torch.cat((left, right), dim=-1)
        return data

class StateEliminationNNet(nn.Module):
    def __init__(self):
        super(StateEliminationNNet, self).__init__()
        self.left_right_gnn = LeftRightGNN()

        self.policy_head1 = nn.Linear(512, 256)
        self.policy_head2 = nn.Linear(256, 128)
        self.policy_head3 = nn.Linear(128, 64)
        self.policy_head4 = nn.Linear(64, 32)
        self.policy_head5 = nn.Linear(32, 16)
        self.policy_head6 = nn.Linear(16, 4)

        self.value_head1 = nn.Linear(512, 256)
        self.value_head2 = nn.Linear(256, 128)
        self.value_head3 = nn.Linear(128, 64)
        self.value_head4 = nn.Linear(64, 32)
        self.value_head5 = nn.Linear(32, 16)
        self.value_head6 = nn.Linear(16, 4)

    def forward(self, left_batch, right_batch):
        data = self.left_right_gnn(left_batch, right_batch)
        pi = F.elu(self.policy_head1(data))
        pi = F.elu(self.policy_head2(pi))
        pi = F.elu(self.policy_head3(pi))
        pi = F.elu(self.policy_head4(pi))
        pi = F.elu(self.policy_head5(pi))
        pi = self.policy_head6(pi)
        data = global_mean_pool(pi, left_batch.batch)
        return data

class memNet(nn.Module):
    def __init__(self):
        super(memNet, self).__init__()
        self.graph = StateEliminationNNet()
        self.word = wordNet()
    
    def forward(self, left_graph, right_graph, word):
        graph = self.graph(left_graph, right_graph)
        word = self.word(word)
        return graph, word

lr = 0.001
epoch = 9999999999999999
model = memNet().cuda()
criterion = torch.nn.CosineEmbeddingLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)
minimum_loss = float("inf")
print(f"Train on {len(train_data)}, validate on {len(test_data)} samples.")
for i in range(epoch):
    model.train()
    total_loss = 0
    count = 0
    for fa, word, label in tqdm(train):
        left_graph = fa[0].contiguous().cuda()
        right_graph = fa[1].contiguous().cuda()
        word = word.contiguous().cuda()
        label = label.contiguous().cuda()
        graph, word = model(left_graph, right_graph, word)
        loss = criterion(graph, word, label)
        total_loss += loss.item()
        count += 1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Train - Epoch {i + 1}, loss: {total_loss / count}")
    with torch.no_grad():
        model.eval()
        total_loss = 0
        count = 0
        for fa, word, label in tqdm(test):
            left_graph = fa[0].contiguous().cuda()
            right_graph = fa[1].contiguous().cuda()
            word = word.contiguous().cuda()
            label = label.contiguous().cuda()
            graph, word = model(left_graph, right_graph, word)
            loss = criterion(graph, word, label)
            total_loss += loss.item()
            count += 1
        print(f"Valid - Epoch {i + 1}, loss: {total_loss / count}")
        if total_loss < minimum_loss:
            minimum_loss = total_loss
            torch.save(model.graph.left_right_gnn.state_dict(), f"{total_loss / count}gnn.pth")
