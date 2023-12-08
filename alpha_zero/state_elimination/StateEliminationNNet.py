import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GATv2Conv
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool

from alpha_zero.utils import *

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
        regex = regex[:, -1]
        return regex

class ForwardBackwardGNN(nn.Module):
    def __init__(self, load_pretrained_embedding_with_lstm=False):
        super(ForwardBackwardGNN, self).__init__()
        self.state_id_dim = MAX_STATES + 3
        self.initial_and_final_dim = 2
        self.transition_dim = (MAX_STATES + 3) * 3
        self.in_regex_idx = self.state_id_dim + self.initial_and_final_dim + (MAX_STATES + 3) * 2
        self.out_regex_idx = self.in_regex_idx + self.transition_dim
        self.hidden_vector_dim = self.out_regex_idx + MAX_STATES + 3

        self.embedding_with_lstm = EmbeddingWithLSTM()
        if load_pretrained_embedding_with_lstm:
            self.embedding_with_lstm.load_state_dict(torch.load("0.191.pth"))
        self.lin1 = nn.Linear(LSTM_DIMENSION * 2, 32)
        self.lin2 = nn.Linear(32, 1)

        self.forward_conv1 = GATv2Conv(self.hidden_vector_dim, self.hidden_vector_dim, add_self_loops=False)
        self.backward_conv1 = GATv2Conv(self.hidden_vector_dim, self.hidden_vector_dim, add_self_loops=False)

    def forward(self, forward_graph, backward_graph):
        forward_edge_attr = self.embedding_with_lstm(forward_graph.edge_attr)
        forward_edge_attr = F.relu(self.lin1(forward_edge_attr))
        forward_graph.edge_attr = torch.flatten(F.relu(self.lin2(forward_edge_attr)))

        backward_edge_attr = self.embedding_with_lstm(backward_graph.edge_attr)
        backward_edge_attr = F.relu(self.lin1(backward_edge_attr))
        backward_graph.edge_attr = torch.flatten(F.relu(self.lin2(backward_edge_attr)))

        forward_source_state_id = torch.argmax(forward_graph.x[forward_graph.edge_index[0], :53], dim=-1)
        forward_target_state_id = torch.argmax(forward_graph.x[forward_graph.edge_index[1], :53], dim=-1)
        backward_source_state_id = torch.argmax(backward_graph.x[backward_graph.edge_index[0], :53], dim=-1)
        backward_target_state_id = torch.argmax(backward_graph.x[backward_graph.edge_index[1], :53], dim=-1)

        #in_regex update
        forward_graph.x[forward_graph.edge_index[1], self.in_regex_idx + forward_source_state_id] = forward_graph.edge_attr
        backward_graph.x[backward_graph.edge_index[1], self.in_regex_idx + backward_source_state_id] = backward_graph.edge_attr

        #out_regex update
        forward_graph.x[forward_graph.edge_index[0], self.out_regex_idx + forward_target_state_id] = forward_graph.edge_attr
        backward_graph.x[backward_graph.edge_index[0], self.out_regex_idx + backward_target_state_id] = backward_graph.edge_attr

        forward_graph.x = F.relu(self.forward_conv1(forward_graph.x, forward_graph.edge_index) + forward_graph.x)
        backward_graph.x = F.relu(self.backward_conv1(backward_graph.x, backward_graph.edge_index) + backward_graph.x)
        graph = torch.cat((forward_graph.x, backward_graph.x), dim=-1)
        return graph

class StateEliminationNNet(nn.Module):
    def __init__(self, load_pretrained_embedding_with_lstm=False, load_forward_backward_gnn=False):
        super(StateEliminationNNet, self).__init__()
        self.forward_backward_gnn = ForwardBackwardGNN(load_pretrained_embedding_with_lstm)
        if load_forward_backward_gnn:
            self.forward_backward_gnn.load_state_dict(torch.load("./alpha_zero/state_elimination/state_weight_7.65gnn.pth"))
        #Single: 373
        #Both: 746
        self.lin1 = nn.Linear(746, 512)
        self.lin2 = nn.Linear(512, 256)
        self.lin3 = nn.Linear(256, 128)
        self.lin4 = nn.Linear(128, 64)
        self.lin5 = nn.Linear(64, 32)
        self.lin6 = nn.Linear(32, 1)
        self.value_lin1 = nn.Linear(746, 512)
        self.value_lin2 = nn.Linear(512, 256)
        self.value_lin3 = nn.Linear(256, 128)
        self.value_lin4 = nn.Linear(128, 64)
        self.value_lin5 = nn.Linear(64, 32)
        self.value_lin6 = nn.Linear(32, 1)

    def forward(self, forward_graph, backward_graph=None):
        #final_node_index = torch.flatten(torch.nonzero(forward_graph.x[:, self.forward_backward_gnn.state_id_dim + 1]))
        graph = self.forward_backward_gnn(forward_graph, backward_graph)
        final_nodes = global_add_pool(graph, forward_graph.batch)
        #final_nodes = graph[final_node_index]
        final_nodes = F.relu(self.value_lin1(final_nodes))
        final_nodes = F.relu(self.value_lin2(final_nodes))
        final_nodes = F.relu(self.value_lin3(final_nodes))
        final_nodes = F.relu(self.value_lin4(final_nodes))
        final_nodes = F.relu(self.value_lin5(final_nodes))
        final_nodes = self.value_lin6(final_nodes)

        graph = F.relu(self.lin1(graph))
        graph = F.relu(self.lin2(graph))
        graph = F.relu(self.lin3(graph))
        graph = F.relu(self.lin4(graph))
        graph = F.relu(self.lin5(graph))
        graph = self.lin6(graph)
        graph = graph.view(-1, 52)
        
        pi = graph
        v = final_nodes

        return F.log_softmax(pi, dim=1), v
