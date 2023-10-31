import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.pool import global_mean_pool

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
        return regex

class StateEliminationNNet(nn.Module):
    def __init__(self, game):
        super(StateEliminationNNet, self).__init__()
        self.action_size = game.getActionSize()
        self.state_number_dim = MAX_STATES + 3
        self.lstm_dim = LSTM_DIMENSION

        self.embedding_with_lstm = EmbeddingWithLSTM()
        self.embedding_with_lstm.load_state_dict(torch.load("./alpha_zero/state_elimination/0.191.pth"))
        for param in self.embedding_with_lstm.parameters():
            param.requires_grad = False

        assert NUMBER_OF_CHANNELS % NUMBER_OF_HEADS == 0
        self.conv1 = GATv2Conv(self.state_number_dim * 3 + self.lstm_dim * 2 * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2)
        self.conv2 = GATv2Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2)
        self.conv3 = GATv2Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2)

        self.policy_head1 = nn.Linear(256, 128)
        self.policy_head2 = nn.Linear(128, 64)
        self.policy_head3 = nn.Linear(64, 32)
        self.policy_head4 = nn.Linear(32, 16)
        self.policy_head5 = nn.Linear(16, 8)
        self.policy_head6 = nn.Linear(8, 4)
        self.policy_head7 = nn.Linear(4, 2)
        self.policy_head8 = nn.Linear(2, 1)

        self.value_head1 = nn.Linear(256, 128)
        self.value_head2 = nn.Linear(128, 64)
        self.value_head3 = nn.Linear(64, 32)
        self.value_head4 = nn.Linear(32, 1)

    def forward(self, data):
        regex = data.edge_attr[:, :MAX_LEN]
        source_state_numbers = data.edge_attr[:, MAX_LEN:MAX_LEN + MAX_STATES + 3]
        target_state_numbers = data.edge_attr[:, MAX_LEN + MAX_STATES + 3:]

        regex = self.embedding_with_lstm(regex)
        regex = regex.mean(1)

        data.edge_attr = regex

        source_states = data.edge_index[0]
        target_states = data.edge_index[1]

        out_transitions = global_mean_pool(torch.cat((target_state_numbers, regex), dim=-1), source_states, data.x.size()[0])
        in_transitions = global_mean_pool(torch.cat((source_state_numbers, regex), dim=-1), target_states, data.x.size()[0])
        data.x = torch.cat((data.x, in_transitions, out_transitions), dim=-1)

        data.x = F.elu(self.conv1(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr))
        data.x = F.elu(self.conv2(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr) + data.x)
        data.x = F.elu(self.conv3(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr) + data.x)

        s = global_mean_pool(data.x, data.batch)
        v = F.elu(self.value_head1(s))
        v = F.elu(self.value_head2(v))
        v = F.elu(self.value_head3(v))
        v = self.value_head4(v)

        pi = F.elu(self.policy_head1(data.x))
        pi = F.elu(self.policy_head2(pi))
        pi = F.elu(self.policy_head3(pi))
        pi = F.elu(self.policy_head4(pi))
        pi = F.elu(self.policy_head5(pi))
        pi = F.elu(self.policy_head6(pi))
        pi = F.elu(self.policy_head7(pi))
        pi = self.policy_head8(pi)
        pi = pi.view(-1, self.action_size)

        '''
        new_x = torch.full((data.batch.max().item() + 1, self.action_size), -999.0).cuda()
        prev, idx = -1, -1
        for i in range(len(data.batch)):
            graph_index = data.batch[i]
            if graph_index != prev:
                prev = graph_index
                idx = 0
            new_x[graph_index][idx] = pi[i]
            idx += 1
        '''

        return F.log_softmax(pi, dim=1), v
