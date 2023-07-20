import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.utils import *

from config import *

class StateEliminationNNet(nn.Module):
    def __init__(self, game):
        self.board_x, self.board_y = game.getBoardSize()
        self.action_size = game.getActionSize()
        super(StateEliminationNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(NUMBER_OF_CHANNELS, NUMBER_OF_CHANNELS, 5, stride=1, padding=2)
        self.fc1 = nn.Linear(NUMBER_OF_CHANNELS * self.board_x * self.board_y, 256)
        self.fc2 = nn.Linear(256, 128)
        self.policy_fc1 = nn.Linear(128, 128)
        self.policy_fc2 = nn.Linear(128, 32)
        self.policy_fc3 = nn.Linear(32, self.action_size)
        self.value_fc1 = nn.Linear(128, 128)
        self.value_fc2 = nn.Linear(128, 32)
        self.value_fc3 = nn.Linear(32, 1)

    def forward(self, s):
        s = s.view(-1, 1, self.board_x, self.board_y)
        s = F.relu((self.conv1(s)))
        s = F.relu((self.conv2(s)))
        s = F.relu((self.conv3(s)))
        s = F.relu((self.conv4(s)))
        s = s.view(-1, NUMBER_OF_CHANNELS * self.board_x * self.board_y)
        s = F.dropout(F.relu((self.fc1(s))), p=DROUPOUT, training=self.training)
        s = F.dropout(F.relu((self.fc2(s))), p=DROUPOUT, training=self.training)
        pi = F.relu(self.policy_fc1(s))
        pi = F.relu(self.policy_fc2(pi))
        pi = self.policy_fc3(pi)
        v = F.relu(self.value_fc1(s))
        v = F.relu(self.value_fc2(v))
        v = self.value_fc3(v)
        return F.log_softmax(pi, dim=1), v
