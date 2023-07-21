import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.utils import *

from config import *

class StateEliminationNNet(nn.Module):
    def __init__(self, game):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        super(StateEliminationNNet, self).__init__()
        self.embedding_dim = EMBEDDING_DIMENSION
        self.embed = nn.Embedding(VOCAB_SIZE, EMBEDDING_DIMENSION)
        self.lstm_dim = LSTM_DIMENSION
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True)
        self.conv1 = nn.Conv2d(self.lstm_dim + 1, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.policy_fc1 = nn.Linear(NUMBER_OF_CHANNELS * self.action_size * 2, 256)
        self.policy_fc2 = nn.Linear(256, 32)
        self.policy_fc3 = nn.Linear(32, 1)
        self.value_fc1 = nn.Linear(NUMBER_OF_CHANNELS * self.action_size * 2, 256)
        self.value_fc2 = nn.Linear(256, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, s):
        #batch * board_x * board_y * (MAX_LEN + tree_length)
        s = s.view(-1, self.board_x, self.board_y, MAX_LEN + 1)
        #(batch * board_x * board_y) * MAX_LEN
        s_re = s[:, :, :, 1:].view(-1, MAX_LEN)
        #batch * 1 * board_x * board_y
        s_len = s[:, :, :, 0].view(-1, 1, self.board_x, self.board_y)
        #(batch * board_x * board_y) * MAX_LEN * EMBEDDING_DIMENSION
        s_re = self.embed(s_re)
        #(batch * board_x * board_y) * LSTM_DIMENSION (get the last time step)
        s_re = self.lstm(s_re)[0][:, -1]
        #batch * LSTM_DIMENSION * board_x * board_y
        s_re = s_re.view(-1, self.board_x, self.board_y, self.lstm_dim).transpose(1, 3).transpose(2, 3)
        #batch * (LSTM_DIMENSION + 1) * board_x * board_y
        s = torch.cat([s_re, s_len], dim=1)
        s = F.relu((self.conv1(s)))
        s = F.relu((self.conv2(s)))
        s = F.relu((self.conv3(s)))
        s = F.relu((self.conv4(s)))
        #batch * NUMBER_OF_CHANNELS * board_x * board_y
        s_conved = s
        tensor_list = []
        for i in range(self.action_size):
            tensor_list.append(torch.cat([s_conved[:, :, i, :], s_conved[:, :, :, i]], dim=-1))
        #batch_size * action_size * NUMBER_OF_CHANNELS * (action_size * 2)
        s_conved = torch.stack(tensor_list, dim=1).view(-1, self.action_size, NUMBER_OF_CHANNELS * self.action_size * 2)
        #batch_size * action_size
        pi = F.relu(self.policy_fc1(s_conved))
        pi = F.relu(self.policy_fc2(pi))
        pi = self.policy_fc3(pi).view(-1, self.action_size)
        #mean pooling
        s_conved_value = s_conved.mean(dim=1)
        v = F.relu(self.value_fc1(s_conved_value))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        return F.log_softmax(pi, dim=1), v