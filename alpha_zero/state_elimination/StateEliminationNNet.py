import torch
import torch.nn as nn
import torch.nn.functional as F
from alpha_zero.state_elimination.gatv3 import GATv3Conv
#from torch_geometric.nn.norm import BatchNorm
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
        #for param in self.embedding_with_lstm.parameters():
        #    param.requires_grad = True
        
        assert NUMBER_OF_CHANNELS % NUMBER_OF_HEADS == 0
        self.conv1 = GATv3Conv(self.state_number_dim * 3 + self.lstm_dim * 2 * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        #self.conv1 = GATv3Conv(2, NUMBER_OF_CHANNELS, heads=1, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False, include_edge_attr=True)
        self.conv2 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv3 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv4 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.conv5 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)

        self.right_conv1 = GATv3Conv(self.state_number_dim * 3 + self.lstm_dim * 2 * 2 + 2, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        #self.right_conv1 = GATv3Conv(2, NUMBER_OF_CHANNELS, heads=1, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False, include_edge_attr=True)
        self.right_conv2 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv3 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv4 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)
        self.right_conv5 = GATv3Conv(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS // NUMBER_OF_HEADS, heads=NUMBER_OF_HEADS, edge_dim=LSTM_DIMENSION * 2, add_self_loops=False)

        self.policy_head1 = nn.Linear(512, 256)
        self.policy_head2 = nn.Linear(256, 128)
        self.policy_head3 = nn.Linear(128, 64)
        self.policy_head4 = nn.Linear(64, 32)
        self.policy_head5 = nn.Linear(32, 16)
        self.policy_head6 = nn.Linear(16, 1)

        self.value_head1 = nn.Linear(512, 256)
        self.value_head2 = nn.Linear(256, 128)
        self.value_head3 = nn.Linear(128, 64)
        self.value_head4 = nn.Linear(64, 32)
        self.value_head5 = nn.Linear(32, 16)
        self.value_head6 = nn.Linear(16, 1)



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
        #batch * 512

        #value는 전혀 학습하지 못 한다.
        s = global_mean_pool(data, left_batch.batch)
        v = F.elu(self.value_head1(s))
        v = F.elu(self.value_head2(v))
        v = F.elu(self.value_head3(v))
        v = F.elu(self.value_head4(v))
        v = F.elu(self.value_head5(v))
        v = self.value_head6(v)

        #policy는 0.6에서 학습이 멈춘다.
        pi = F.elu(self.policy_head1(data))
        pi = F.elu(self.policy_head2(pi))
        pi = F.elu(self.policy_head3(pi))
        pi = F.elu(self.policy_head4(pi))
        pi = F.elu(self.policy_head5(pi))
        pi = self.policy_head6(pi)
        pi = pi.view(-1, self.action_size)

        return F.log_softmax(pi, dim=1), v
