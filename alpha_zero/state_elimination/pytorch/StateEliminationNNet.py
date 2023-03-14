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
        
        self.embedding_dim = args.embedding_dim
        
        self.embed = nn.Embedding(args.vocab_size, args.embedding_dim)
        
        self.lstm = nn.LSTM(args.embedding_dim,
                            args.num_channels, batch_first=True)
        
        self.conv1 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=0)
        self.conv3 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=0)
        self.conv4 = nn.Conv2d(
            args.num_channels, args.num_channels, 3, stride=1, padding=0)
        
        self.bn1 = nn.BatchNorm2d(args.num_channels)
        self.bn2 = nn.BatchNorm2d(args.num_channels)
        self.bn3 = nn.BatchNorm2d(args.num_channels)
        self.bn4 = nn.BatchNorm2d(args.num_channels)
        self.fc1 = nn.Linear(
            args.num_channels*(self.board_x - 8)*(self.board_y - 8), 256)
        self.fc_bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(512, 512)
        #self.fc4 = nn.Linear(256, 128)
        self.fc_bn2 = nn.BatchNorm1d(128)
        self.policy_fc1 = nn.Linear(128, 128)
        self.policy_fc2 = nn.Linear(128, 32)
        self.policy_fc3 = nn.Linear(32, self.action_size)
        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, s):
        #                                                           s: batch_size x board_x x board_y
        # (batch_size x board_x x board_y) x 20
        s = s.view(-1, 20)
        # (batch_size x board_x x board_y) x 20 x embedding_dim
        s = self.embed(s)
        # (batch_size x board_x x board_y) x 20 x num_channels
        s, _ = self.lstm(s)
        
        s = s[:, -1].view(-1, self.board_x, self.board_y, self.args.num_channels).transpose(1, 3).transpose(2, 3)
        
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv1(s)))
        # batch_size x num_channels x board_x x board_y
        s = F.relu((self.conv2(s)))
        # batch_size x num_channels x (board_x-2) x (board_y-2)
        s = F.relu((self.conv3(s)))
        # batch_size x num_channels x (board_x-4) x (board_y-4)
        s = F.relu((self.conv4(s)))
        
        s = s.reshape(-1, self.args.num_channels *
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
