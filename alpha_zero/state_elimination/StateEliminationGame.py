from math import log

import torch
import numpy as np
from FAdo.reex import *
from torch_geometric.data import Data

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization

from config import *

#0 as emptyness, 5 + max(ALPHABET) should be input size of embedding dimension
word_to_ix = {'@': 1, '+': 2, '*': 3, '.': 4}
for i in range(max(ALPHABET)):
    word_to_ix[str(i)] = i + 5

class StateEliminationGame():
    def __init__(self, maxN=MAX_STATES):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n, k, d = 5, 5, 0.1
            gfa = generate(n, k, d)
        self.n, self.k, self.d = n, k, d
        return gfa

    def get_encoded_regex(self, regex):
        #NB: This technique cannot be applied to GFA with an alphabet size of more than 9.
        encoded_regex = [word_to_ix[word] for word in list(regex.rpn().replace("@epsilon", "@").replace("'", ""))[:MAX_LEN]]
        if len(encoded_regex) < MAX_LEN:
            encoded_regex = encoded_regex + [0] * (MAX_LEN - len(encoded_regex))
        assert len(encoded_regex) == MAX_LEN
        return encoded_regex

    def gfa_to_tensor(self, gfa):
        num_nodes = len(gfa.States)
        x = []
        edge_index = [[], []]
        edge_attr = []
        for source in gfa.delta:
            source_state_number = int(gfa.States[source])
            is_initial_state = 1 if source == gfa.Initial else 0
            is_final_state = 1 if source in gfa.Final else 0
            x.append([source_state_number, is_initial_state, is_final_state])
            for target in gfa.delta[source]:
                target_state_number = int(gfa.States[target])
                edge_index[0].append(source)
                edge_index[1].append(target)
                edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]) + [source_state_number, target_state_number])
        x = torch.LongTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.LongTensor(edge_attr)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        return graph

    def getActionSize(self):
        return self.maxN + 2

    def getNextState(self, gfa, action, duplicate=False, minimize=False):
        final_state = list(gfa.Final)[0]
        assert 0 < action < final_state
        if duplicate:
            eliminated_gfa = eliminate_with_minimization(gfa.dup(), action, minimize=minimize)
        else:
            eliminated_gfa = eliminate_with_minimization(gfa, action, minimize=minimize)
        return eliminated_gfa

    def getValidMoves(self, gfa):
        final_state = list(gfa.Final)[0]
        validMoves = [0 for i in range(self.maxN + 2)]
        for i in range(1, final_state):
            validMoves[i] = 1
        return validMoves
    
    def get_resulting_regex(self, gfa):
        result = CConcat(CConcat(gfa.delta[0][1], CStar(gfa.delta[1][1])), gfa.delta[1][2]) if 1 in gfa.delta[1] else CConcat(gfa.delta[0][1], gfa.delta[1][2])
        result = CDisj(gfa.delta[0][2], result) if 2 in gfa.delta[0] else result
        return result

    def getGameEnded(self, gfa):
        if len(gfa.States) == 3:
            result = self.get_resulting_regex(gfa)
            length = result.treeLength()
            reward = -log(length)
            return reward
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
