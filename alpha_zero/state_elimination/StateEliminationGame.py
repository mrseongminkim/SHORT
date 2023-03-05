import numpy as np

from alpha_zero.Game import Game
from utils.random_nfa_generator import generate

class StateEliminationGame(Game):
    def __init__(self, maxN = 18):
        self.maxN = maxN

    def getInitBoard(self, gfs, n, k, d):
        self.n = n
        self.k = k
        self.d = d
        self.gfs = gfs.dup()
        #self.n = np.random.randint(5, 11) #np.random.randint(3, maxN)
        #self.k = 5 #np.random.choice([2, 5, 10])
        #self.d = 0.2 #np.random.choice([0.2, 0.5])
        #self.gfs = generate(self.n, self.k, self.d, 'in-memory')
        board = np.zeros((self.maxN + 2, self.maxN + 2), dtype = int)
        for source_state in self.gfs.delta:
            for target_state in self.gfs.delta[source_state]:
                board[source_state][target_state] = self.gfs.delta[source_state][target_state].treeLength()
        # 0 as initial, self.n + 1 as final
        return board
    
    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

    def getActionSize(self):
        return self.maxN

    def getNextState(self, board, player, action):
        new_board = np.copy(board)
        action += 1
        if action >= self.n + 1:
            return (new_board, player)
        self_loop = new_board[action][action]
        punct = 3 if self_loop else 1
        for source_state in range(self.n + 1):
            if new_board[source_state][action]:
                alpha = new_board[source_state][action]
                new_board[source_state][action] = 0
                for target_state in range(self.n + 2):
                    if new_board[action][target_state]:
                        beta = new_board[action][target_state]
                        new_board[source_state][target_state] += alpha + beta + punct + self_loop + (1 if new_board[source_state][target_state] else 0)
        new_board[action] = np.zeros(self.maxN + 2, dtype = int)
        return (new_board, player)

    def getValidMoves(self, board, player):
        validMoves = [0 for i in range(self.maxN)]
        for i in range(1, self.n + 1):
            validMoves[i - 1] = int(np.any(board[i]))
        return validMoves

    def getGameEnded(self, board, player):
        #-1 as not finished, value for transition
        for i in range(1, self.n + 1):
            if np.any(board[i]):
                return -1
        return board[0][self.n + 1] / (3.33) ** (self.n)

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()