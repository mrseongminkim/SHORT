import torch
import numpy as np
from FAdo.reex import *

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization

from config import *

#0 as emptyness, 5 + max(ALPHABET) should be input size of embedding dimension
word_to_ix = {'@': 1, '+': 2, '*': 3, '.': 4}
for i in range(max(ALPHABET)):
    word_to_ix[str(i)] = i + 5

class StateEliminationGame():
    def __init__(self, maxN=50):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n, k, d = 7, 5, 0.1
            gfa = generate(n, k, d)
        return gfa

    def gfa_to_tensor(self, gfa):
        length_board = torch.zeros((self.maxN + 2, self.maxN + 2), dtype=torch.long)
        regex_board = [[[0] * MAX_LEN for i in range(self.maxN + 2)] for i in range(self.maxN + 2)]
        for source in gfa.delta:
            for target in gfa.delta[source]:
                length_board[source][target] = gfa.delta[source][target].treeLength()
                #NB: This technique cannot be applied to GFA with an alphabet size of more than 9.
                encoded_regex = [word_to_ix[word] for word in list(gfa.delta[source][target].rpn().replace("@epsilon", "@").replace("'", ""))[:MAX_LEN]]
                if len(encoded_regex) < MAX_LEN:
                    encoded_regex = encoded_regex + [0] * (MAX_LEN - len(encoded_regex))
                assert len(encoded_regex) == MAX_LEN
                regex_board[source][target] = encoded_regex        
        board = torch.cat((length_board.unsqueeze(2), torch.LongTensor(regex_board)), dim = 2)
        return board

    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

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

    def getGameEnded(self, gfa):
        if len(gfa.States) == 2:
            length = gfa.delta[0][1].treeLength()
            AVG = 328.5862
            STD = 149.44780839578556
            reward = (length - AVG) / STD
            return - reward
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
