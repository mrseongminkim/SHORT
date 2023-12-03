import sys
sys.path.append(r"/home/toc/seongmin")
sys.path.append(r"/home/toc/seongmin/SHORT")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATv2Conv

from SHORT.config import *

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
        regex = regex[:, -1]
        return regex

class ForwardBackwardGNN(nn.Module):
    def __init__(self):
        super(ForwardBackwardGNN, self).__init__()
        self.state_id_dim = MAX_STATES + 3
        self.initial_and_final_dim = 2
        self.connectivity_dim = MAX_STATES + 3
        self.length_dim = MAX_STATES + 3
        self.regex_idx = self.state_id_dim + self.initial_and_final_dim + self.connectivity_dim + self.length_dim
        self.hidden_vector_dim = self.regex_idx + MAX_STATES + 3

        self.embedding_with_lstm = EmbeddingWithLSTM()
        self.embedding_with_lstm.load_state_dict(torch.load("0.191.pth"))
        self.lin1 = nn.Linear(LSTM_DIMENSION * 2, 32)
        self.lin2 = nn.Linear(32, 1)
        self.forward_conv1 = GATv2Conv(self.hidden_vector_dim, self.hidden_vector_dim, add_self_loops=False)
        self.backward_conv1 = GATv2Conv(self.hidden_vector_dim, self.hidden_vector_dim, add_self_loops=False)

    def forward(self, forward_graph, backward_graph=None):
        forward_edge_attr = self.embedding_with_lstm(forward_graph.edge_attr)
        forward_edge_attr = F.relu(self.lin1(forward_edge_attr))
        forward_graph.edge_attr = torch.flatten(F.relu(self.lin2(forward_edge_attr)))

        backward_edge_attr = self.embedding_with_lstm(backward_graph.edge_attr)
        backward_edge_attr = F.relu(self.lin1(backward_edge_attr))
        backward_graph.edge_attr = torch.flatten(F.relu(self.lin2(backward_edge_attr)))

        forward_target_state_id = torch.argmax(forward_graph.x[forward_graph.edge_index[1], :53], dim=-1)
        backward_target_state_id = torch.argmax(backward_graph.x[backward_graph.edge_index[1], :53], dim=-1)

        forward_graph.x[forward_graph.edge_index[0], self.regex_idx + forward_target_state_id] = forward_graph.edge_attr
        backward_graph.x[backward_graph.edge_index[1], self.regex_idx + backward_target_state_id] = backward_graph.edge_attr

        forward_graph = F.relu(self.forward_conv1(forward_graph.x, forward_graph.edge_index) + forward_graph.x)
        backward_graph = F.relu(self.backward_conv1(backward_graph.x, backward_graph.edge_index) + backward_graph.x)
        graph = torch.cat((forward_graph, backward_graph), dim=-1)
        return graph

class StateWeightNet(nn.Module):
    def __init__(self):
        super(StateWeightNet, self).__init__()
        self.forward_backward_gnn = ForwardBackwardGNN()
        self.lin1 = nn.Linear(428, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 1)

    def forward(self, forward_graph, backward_graph=None):
        graph = self.forward_backward_gnn(forward_graph, backward_graph)
        graph = F.relu(self.lin1(graph))
        graph = F.relu(self.lin2(graph))
        graph = F.relu(self.lin3(graph))
        graph = F.relu(self.lin4(graph))
        graph = self.lin5(graph)
        graph = graph.view(-1, 52)
        return graph
