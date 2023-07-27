import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.pool import global_mean_pool

from alpha_zero.utils import *

from config import *

class StateEliminationNNet(nn.Module):
    def __init__(self, game):
        super(StateEliminationNNet, self).__init__()
        self.action_size = game.getActionSize()
        self.state_number_embedding_dim = STATE_NUMBER_EMBEDDING_DIMENSION
        self.regex_embedding_dim = REGEX_EMBEDDING_DIMENSION
        self.state_number_embed = nn.Embedding(self.action_size, self.state_number_embedding_dim)
        self.regex_embed = nn.Embedding(VOCAB_SIZE, self.regex_embedding_dim)
        self.lstm_dim = LSTM_DIMENSION
        self.lstm = nn.LSTM(self.regex_embedding_dim, self.lstm_dim, batch_first=True)



        assert NUMBER_OF_CHANNELS % NUMBER_OF_HEADS == 0
        #75 -> 75
        self.conv1 = GATv2Conv(self.state_number_embedding_dim * 3 + self.lstm_dim * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION)
        self.conv2 = GATv2Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION)
        self.conv3 = GATv2Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION)
        self.conv4 = GATv2Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION)

        self.policy_head1 = nn.Linear(NUMBER_OF_CHANNELS, 32)
        self.policy_head2 = nn.Linear(32, 1)
        self.value_head1 = nn.Linear(NUMBER_OF_CHANNELS, 32)
        self.value_head2 = nn.Linear(32, 1)

    def forward(self, data):
        states = data.x[:, 0]
        additional_information = data.x[:, 1:]
        states = self.state_number_embed(states)
        data.x = torch.cat((states, additional_information), dim=-1)

        source_state_numbers = data.edge_attr[:, -2]
        target_state_numbers = data.edge_attr[:, -1]
        encoded_regex = data.edge_attr[:, :-2]
        source_state_numbers = self.state_number_embed(source_state_numbers)
        target_state_numbers = self.state_number_embed(target_state_numbers)
        encoded_regex = self.regex_embed(encoded_regex)
        encoded_regex = self.lstm(encoded_regex)[0][:, -1]
        data.edge_attr = encoded_regex

        source_states = data.edge_index[0]
        target_states = data.edge_index[1]
        out_transitions = global_mean_pool(torch.cat((target_state_numbers, encoded_regex), dim=-1), source_states, data.x.size()[0])
        in_transitions = global_mean_pool(torch.cat((source_state_numbers, encoded_regex), dim=-1), target_states, data.x.size()[0])
        data.x = torch.cat((data.x, in_transitions, out_transitions), dim=-1)

        data.x = F.relu(self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)) + data.x
        data.x = F.relu(self.conv2(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)) + data.x
        data.x = F.relu(self.conv3(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)) + data.x
        data.x = F.relu(self.conv4(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr)) + data.x

        s = global_mean_pool(data.x, data.batch)
        v = F.relu(self.value_head1(s))
        v = self.value_head2(v)

        pi = F.relu(self.policy_head1(data.x))
        pi = self.policy_head2(pi)
        new_x = torch.full((data.batch.max().item() + 1, self.action_size), -999.0).cuda()
        prev, idx = -1, -1
        for i in range(len(data.batch)):
            graph_index = data.batch[i]
            if graph_index != prev:
                prev = graph_index
                idx = 0
            new_x[graph_index][idx] = pi[i].item()
            idx += 1
        return F.log_softmax(new_x, dim=1), v
#Now optimality of training data is guaranteed, the problem is NN itself.
