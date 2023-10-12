import torch
import torch.nn as nn
import torch.nn.functional as F

from alpha_zero.set_transformer.model import *

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
        self.embedding_with_lstm.load_state_dict(torch.load("./alpha_zero/state_elimination/embed_lstm.pth"))
        self.set_transformer = SetTransformer(12223, NUMBER_OF_CHANNELS)

        self.value_head1 = nn.Linear(256, 128)
        self.value_head2 = nn.Linear(128, 64)
        self.value_head3 = nn.Linear(64, 32)
        self.value_head4 = nn.Linear(32, 1)

        self.policy_head1 = nn.Linear(256, 128)
        self.policy_head2 = nn.Linear(128, 64)
        self.policy_head3 = nn.Linear(64, 32)
        self.policy_head4 = nn.Linear(32, 1)

    def forward(self, data):
        state_number = data[:, :, :self.state_number_dim]

        is_initial = data[:, :, self.state_number_dim].unsqueeze(-1)

        is_final = data[:, :, self.state_number_dim + 1].unsqueeze(-1)

        in_transition_label = data[:, :, 55:55 + 50 * 52]
        in_transition_label = in_transition_label.reshape(-1, MAX_LEN)
        in_transition_label = self.embedding_with_lstm(in_transition_label)
        in_transition_label = in_transition_label.mean(1)

        in_transition_state = data[:, :, 55 + 50 * 52:55 + 50 * 52 + 53 * 52]
        in_transition_state = in_transition_state.reshape(-1, self.state_number_dim)

        out_transition_label = data[:, :, 55 + 50 * 52 + 53 * 52:55 + 50 * 52 + 53 * 52 + 50 * 52]
        out_transition_label = out_transition_label.reshape(-1, MAX_LEN)
        out_transition_label = self.embedding_with_lstm(out_transition_label)
        out_transition_label = out_transition_label.mean(1)

        out_transition_state = data[:, :, 55 + 50 * 52 + 53 * 52 + 50 * 52:]
        out_transition_state = out_transition_state.reshape(-1, self.state_number_dim)

        in_transition = torch.cat((in_transition_label, in_transition_state), -1)
        in_transition = in_transition.reshape(-1, 52, 52 * 117)
        out_transition = torch.cat((out_transition_label, out_transition_state), -1)
        out_transition = out_transition.reshape(-1, 52, 52 * 117)

        data = torch.cat((state_number, is_initial, is_final, in_transition, out_transition), dim=-1)
        value, policy = self.set_transformer(data)

        value = F.relu(self.value_head1(value))
        value = F.relu(self.value_head2(value))
        value = F.relu(self.value_head3(value))
        value = self.value_head4(value)

        policy = F.relu(self.policy_head1(policy))
        policy = F.relu(self.policy_head2(policy))
        policy = F.relu(self.policy_head3(policy))
        policy = self.policy_head4(policy)

        return F.log_softmax(policy.squeeze(-1), dim=1), value
