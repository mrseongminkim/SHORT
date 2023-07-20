import numpy as np
from FAdo.reex import *

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization

from config import *

class StateEliminationGame():
    def __init__(self, maxN=50):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n, k, d = 7, 5, 0.1
            gfa = generate(n, k, d)
        return gfa

    def gfa_to_tensor(self, gfa):
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
            AVG = 306.4036842105266
            STD = 146.21107624649574
            reward = (length - AVG) / STD
            return - reward
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
