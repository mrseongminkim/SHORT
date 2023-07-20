import logging
import math

import numpy as np

from config import *

log = logging.getLogger(__name__)

class MCTS():
    def __init__(self, game, nnet):
        self.game = game
        self.nnet = nnet
        self.Qsa = {}
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}

    def getActionProb(self, gfa, temp=1):
        for i in range(NUMBER_OF_MCTS_SIMULATIONS):
            self.search(gfa)
        s = self.game.stringRepresentation(gfa)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs
        counts = [x ** (1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, gfa):
        s = self.game.stringRepresentation(gfa)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(gfa)
        if self.Es[s] != None:
            return self.Es[s]
        if s not in self.Ps:
            gfa_representation = self.game.gfa_to_tensor(gfa)
            self.Ps[s], v = self.nnet.predict(gfa_representation)
            valids = self.game.getValidMoves(gfa)
            self.Ps[s] = np.exp(self.Ps[s]) * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])
            self.Vs[s] = valids
            self.Ns[s] = 0
            return v
        valids = self.Vs[s]
        cur_best = -float('inf')
        for a in range(self.game.getActionSize()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + self.Nsa[(s, a)])
                else:
                    u = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        next_gfa = self.game.getNextState(gfa, a, duplicate=True)
        v = self.search(next_gfa)
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return v
