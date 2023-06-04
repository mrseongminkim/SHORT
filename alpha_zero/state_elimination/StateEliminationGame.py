from copy import copy
import sys

import numpy as np
from FAdo.reex import *

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization
from utils.fadomata import shuffle_gfa

EPS = 1e-8

class StateEliminationGame():
    def __init__(self, maxN=50):
        self.maxN = maxN

    def getInitBoard(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n = 7
            k = 5
            d = 0.1
            gfa = generate(n, k, d, 'in-memory')
        #self.n = n + 2
        #self.k = k
        #self.d = d
        return gfa

    def gfaToBoard(self, gfa):
        self.alphabet = copy.copy(gfa.Sigma)
        board = np.zeros((self.maxN + 2, self.maxN + 2), dtype=int)
        #re_board = [['' for i in range(self.maxN + 2)] for i in range(self.maxN + 2)]
        for source in gfa.delta:
            for target in gfa.delta[source]:
                board[source][target] = gfa.delta[source][target].treeLength()
                #re_board[source][target] = gfa.delta[source][target]
        return board#, re_board

    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

    def getActionSize(self):
        return self.maxN + 2

    def getNextState(self, gfa, player, action, duplicate=False, minimize=True):
        final_state = list(gfa.Final)[0]
        assert 0 < action and action < final_state
        if duplicate:
            gfa_eliminated = eliminate_with_minimization(gfa.dup(), action, minimize=minimize)
        else:
            gfa_eliminated = eliminate_with_minimization(gfa, action, minimize=minimize)
        return (gfa_eliminated, player)

    def getValidMoves(self, gfa, player):
        final_state = list(gfa.Final)[0]
        validMoves = [0 for i in range(self.maxN + 2)]
        for i in range(1, final_state):
            validMoves[i] = 1
        return validMoves

    def getGameEnded(self, gfa, player):
        if len(gfa.States) == 2:
            length = gfa.delta[0][1].treeLength()
            AVG = 306.4036842105266
            STD = 146.21107624649574
            reward = (length - AVG) / STD
            return - reward
        else:
            return None

    def getCanonicalForm(self, gfa, player):
        return gfa

    def getSymmetries(self, gfa, pi):
        return [(gfa, pi)]

    def stringRepresentation(self, gfa):
        board = self.gfaToBoard(gfa)
        return board.tostring()
