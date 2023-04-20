import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.utils import *


class StateEliminationNNet(nn.Module):
    def __init__(self, game, args):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        self.args = args
        super(StateEliminationNNet, self).__init__()

        #regex_board
        self.embedding_dim = args.embedding_dim
        self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        self.lstm_dim = 32
        
        #regex_board
        self.lstm = nn.LSTM(args.embedding_dim,
                            self.lstm_dim, batch_first=True)
        
        self.conv1 = nn.Conv2d(
            self.lstm_dim + 1, args.num_channels, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 5, stride=1, padding=2)
        
        self.fc1 = nn.Linear(
            args.num_channels*(self.action_size)*(self.action_size), 256)
        self.fc2 = nn.Linear(256, 256)
        self.policy_fc1 = nn.Linear(
            self.args.num_channels * self.action_size * 2, 256)
        self.policy_fc2 = nn.Linear(256, 32)
        self.policy_fc3 = nn.Linear(32, 1)
        self.value_fc1 = nn.Linear(
            self.args.num_channels * self.action_size * 2, 256)
        self.value_fc2 = nn.Linear(256, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # (batch_size x board_x x board_y) x 20
        
        s = s.view(-1, self.board_x, self.board_y, self.args.re_len + 1)
        
        s = s[:, :self.action_size, : self.action_size]
                
        s_re = s[:, :, :, 1:].view(-1, self.args.re_len)
        s_len = s[:, :, :, 0].view(-1, 1, self.action_size, self.action_size)
        # (batch_size x board_x x board_y) x 20 x embedding_dim
        s_re = self.embed(s_re)
        # (batch_size x board_x x board_y) x 20 x num_channels
        s_re, _ = self.lstm(s_re)
        
        s_re = s_re[:, -1].view(-1, self.action_size, self.action_size,
                             self.lstm_dim).transpose(1, 3).transpose(2, 3)
        
        s = torch.cat([s_re, s_len], dim=1)
        
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu((self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu((self.conv4(s)))
        
        s_conved = s
        
        tensor_list = []
        for i in range(self.action_size):
            tensor_list.append(torch.cat([s_conved[:, :, i, :self.action_size], s_conved[:, :, :self.action_size, i]], dim=-1))
            
        # batch_size x action_size x num_channels x (action_size * 2)
        s_conved = torch.stack(tensor_list, dim=1).view(-1, self.action_size,
                                                        self.args.num_channels * self.action_size * 2)
        
        # batch_size x action_size
        pi = F.relu(self.policy_fc1(s_conved))
        pi = F.relu(self.policy_fc2(pi))
        pi = self.policy_fc3(pi).view(-1, self.action_size)
        # batch_size x 1
        
        s_conved_value = s_conved.mean(dim=1)
        
        v = F.relu(self.value_fc1(s_conved_value))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        return F.log_softmax(pi, dim=1), v
