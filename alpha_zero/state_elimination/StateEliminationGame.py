import numpy as np

from alpha_zero.Game import Game
from utils.random_nfa_generator import generate


class StateEliminationGame(Game):
    def __init__(self, maxN=18):
        self.maxN = maxN

    def getInitBoard(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n = np.random.randint(5, 11)  # np.random.randint(3, maxN)
            k = 5  # np.random.choice([2, 5, 10])
            d = 0.2  # np.random.choice([0.2, 0.5])
            gfa = generate(n, k, d, 'in-memory')
        self.n = n + 2
        self.k = k
        self.d = d
        self.gfa = gfa.dup()  # it could be overhead
        board = np.zeros((self.maxN + 2, self.maxN + 2), dtype=int)
        for source_state in self.gfa.delta:
            for target_state in self.gfa.delta[source_state]:
                board[source_state][target_state] = self.gfa.delta[source_state][target_state].treeLength()
        # 0 as initial, self.n - 1 as final
        return board

    def getBoardSize(self):
        return (self.maxN + 2, self.maxN + 2)

    def getActionSize(self):
        return self.maxN

    def getNextState(self, board, player, action):
        new_board = np.copy(board)
        #action += 1
        
        assert action < self.n - 1 and action > 0
        #    return (new_board, player)
        self_loop = new_board[action][action]
        punct = 3 if self_loop else 1
        
        for source_state in range(self.n - 1):
            if new_board[source_state][action] and source_state != action:
                alpha = new_board[source_state][action]
                new_board[source_state][action] = 0
                for target_state in range(self.n):
                    if new_board[action][target_state] and target_state != action:
                        beta = new_board[action][target_state]
                        new_board[source_state][target_state] += alpha + self_loop + \
                            beta + punct + \
                            (1 if new_board[source_state][target_state] else 0)
        new_board[action] = np.zeros(self.maxN + 2, dtype=int)

        assert sum(new_board[action]) == 0
        assert sum(new_board[:, action] == 0)
        
        return (new_board, player)

    def getValidMoves(self, board, player):
        validMoves = [0 for i in range(self.maxN)]
        for i in range(1, self.n-1):
            validMoves[i] = int(np.any(board[i]))
        return validMoves

    def getGameEnded(self, board, player):
        # -1 as not finished, value for transition
        
        # print(board[1:self.n - 1, 1:self.n - 1].sum())
        
        if board[1:self.n - 1, 1:self.n - 1].sum() != 0:
            return -1
        return (1 / board[0][self.n - 1]) * 2 - 1

    def getCanonicalForm(self, board, player):
        return board

    def getSymmetries(self, board, pi):
        return [(board, pi)]

    def stringRepresentation(self, board):
        return board.tostring()
