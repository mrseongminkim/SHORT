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
        self.conv1 = nn.Conv2d(1, args.num_channels, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(args.num_channels, args.num_channels, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(args.num_channels, args.num_channels, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(args.num_channels, args.num_channels, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(args.num_channels * (self.board_x - 8) * (self.board_y - 8), 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_fc1 = nn.Linear(128, 128)
        self.policy_fc2 = nn.Linear(128, 32)
        self.policy_fc3 = nn.Linear(32, self.action_size)
        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # batch_size x 1 x board_x x board_y
        s = s.view(-1, 1, self.board_x, self.board_y)
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu((self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu((self.conv4(s)))
        
        s = s.view(-1, self.args.num_channels *
                   (self.board_x - 8)*(self.board_y - 8))
        s = F.dropout(F.relu((self.fc1(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 1024
        s = F.dropout(F.relu((self.fc2(s))), p=self.args.dropout,
                      training=self.training)  # batch_size x 512
        #s = F.dropout(F.relu((self.fc3(s))), p=self.args.dropout,
        #              training=self.training)  # batch_size x 512
        #s = F.dropout(F.relu((self.fc4(s))), p=self.args.dropout,
        #              training=self.training)  # batch_size x 512
        # batch_size x action_size
        pi = F.relu(self.policy_fc1(s))
        pi = F.relu(self.policy_fc2(pi))
        pi = self.policy_fc3(pi)
        # batch_size x 1
        v = F.relu(self.value_fc1(s))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        return F.log_softmax(pi, dim=1), v
