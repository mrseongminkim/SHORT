import numpy as np

from alpha_zero.Game import Game
from utils.random_nfa_generator import generate
from FAdo.reex import *
from copy import copy, deepcopy

from utils.heuristics import eliminate_new

EPS = 1e-8


class StateEliminationGame(Game):
    def __init__(self, maxN=10):
        self.maxN = maxN

    def getInitBoard(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n = np.random.randint(3, 8)  # np.random.randint(3, maxN)
            k = 2  # np.random.choice([2, 5, 10])
            d = np.random.choice([0.2])
            gfa = generate(n, k, d, 'in-memory')
        self.n = n + 2
        self.k = k
        self.d = d
        self.gfa = gfa.dup()  # it could be overhead
        return gfa
    
    def gfaToBoard(self, gfa):
        
        gfa_dup = gfa.dup()
        
        self.alphabet = copy(gfa_dup.Sigma)
        board = np.zeros((self.maxN + 2, self.maxN + 2), dtype=int)
        re_board = [['' for i in range(self.maxN + 2)] for i in range(self.maxN + 2)]
        for source in gfa_dup.delta:
            for target in gfa_dup.delta[source]:
                board[source][target] = gfa_dup.delta[source][target].treeLength()
                re_board[source][target] = gfa_dup.delta[source][target]
        return re_board

    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

    def getActionSize(self):
        return self.maxN

    def getNextState(self, gfa, player, action):
        # new_board = np.copy(board)
        # action += 1
        
        assert action < self.n - 1 and action > 0
        #    return (new_board, player)
        # self_loop = new_board[action][action]
        # punct = 3 if self_loop else 1
        
        gfa_eliminated = eliminate_new(gfa.dup(), action)
        
        # assert sum(new_board[action]) == 0
        # assert sum(new_board[:, action] == 0)
        
        return (gfa_eliminated, player)

    def getValidMoves(self, gfa, player):
        
        board = self.gfaToBoard(gfa)
        
        validMoves = [0 for i in range(self.maxN)]
        for i in range(1, self.n-1):
            validMoves[i] = int(sum([len(str(word)) for word in board[i]]) > 0)
            
        return validMoves

    def getGameEnded(self, gfa, player):
        # -1 as not finished, value for transition
        
        board = self.gfaToBoard(gfa)
        
        sum_length = 0
        for i in range(1, self.n - 1):
            for j in range(self.maxN + 2):
                sum_length += board[i][j].treeLength() if board[i][j] else 0
        
        if sum_length != 0:
            return -1
        
        return - board[0][self.n - 1].treeLength() / (4) ** (self.n - 2) + EPS

    def getCanonicalForm(self, gfa, player):
        return gfa

    def getSymmetries(self, gfa, pi):
        return [(gfa, pi)]

    def stringRepresentation(self, gfa):
        
        board = self.gfaToBoard(gfa)
        
        len_board = [[re.treeLength() if re else 0 for re in line] for line in board]
        # ' '.join([' '.join([str(re) for re in line]) for line in board])
        return np.array(len_board).tostring()
