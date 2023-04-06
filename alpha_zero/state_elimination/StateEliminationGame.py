import numpy as np

from utils.random_nfa_generator import generate
from FAdo.reex import *
from copy import copy, deepcopy

from utils.heuristics import eliminate_with_minimization

EPS = 1e-8

class StateEliminationGame():
    def __init__(self, maxN=10):
        self.maxN = maxN

    def getInitBoard(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n = 7 # np.random.randint(3, 4)  # np.random.randint(3, maxN)
            k = 5  # np.random.choice([2, 5, 10])
            d = np.random.choice([0.2])
            gfa = generate(n, k, d, 'in-memory')
        self.n = n + 2
        self.k = k
        self.d = d
        return gfa
    
    def gfaToBoard(self, gfa):
        self.alphabet = copy(gfa.Sigma)
        board = np.zeros((self.maxN + 2, self.maxN + 2), dtype=int)
        re_board = [['' for i in range(self.maxN + 2)] for i in range(self.maxN + 2)]
        for source in gfa.delta:
            for target in gfa.delta[source]:
                board[source][target] = gfa.delta[source][target].treeLength()
                re_board[source][target] = gfa.delta[source][target]
        return board, re_board

    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

    def getActionSize(self):
        return self.maxN + 2

    def getNextState(self, gfa, player, action, duplicate=False):
        final_state = list(gfa.Final)[0]
        assert 0 < action and action < final_state
        if duplicate:
            gfa_eliminated = eliminate_with_minimization(gfa.dup(), action)
        else:
            gfa_eliminated = eliminate_with_minimization(gfa, action)
        return (gfa_eliminated, player)

    def getValidMoves(self, gfa, player):
        final_state = list(gfa.Final)[0]
        validMoves = [0 for i in range(self.maxN + 2)]
        for i in range(1, final_state):
            validMoves[i] = 1
        return validMoves

    def getGameEnded(self, gfa, player):
        # -1 as not finished, value for transition
        if len(gfa.States) == 2:
            return - gfa.delta[0][1].treeLength() / (4) ** (self.n - 2) + EPS
        else:
            return -1

    def getCanonicalForm(self, gfa, player):
        return gfa

    def getSymmetries(self, gfa, pi):
        return [(gfa, pi)]

    def stringRepresentation(self, gfa):
        board, re_board = self.gfaToBoard(gfa)
        return board.tostring()
        #len_board = [[re.treeLength() if re else 0 for re in line] for line in board]
        # ' '.join([' '.join([str(re) for re in line]) for line in board])
        #return np.array(len_board).tostring()
