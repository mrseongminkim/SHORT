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
        self.dead_end = set()
        self.actual_reward = set()

    def normalize(self, q, q_max, q_min):
        return (q - q_min) / (q_max - q_min + EPS)

    def getActionProb(self, gfa, temp=0):
        for _ in range(NUMBER_OF_MCTS_SIMULATIONS):
            _, dead_end, _ = self.search(gfa)
            if dead_end: break
        s = self.game.stringRepresentation(gfa)
        #visits = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.getActionSize())]
        action_space = [self.Qsa[(s, a)] for a in range(self.game.getActionSize()) if (s, a) in self.Qsa]
        q_max = max(action_space)
        q_min = min(action_space)
        #print("q_max:", q_max)
        #print("q_min:", q_min)
        #print("visits:", visits)
        #print("action_space:", action_space)

        if q_max == q_min:
            counts = [1 if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
            temp = 1
        else:
            counts = [self.normalize(self.Qsa[(s, a)], q_max, q_min) + 1 if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())]

        if temp == 0 and q_max != q_min:
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
        #returns reward, is_dead_end, is_actual_reward
        s = self.game.stringRepresentation(gfa)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(gfa)
        if self.Es[s] != None:
            return self.Es[s], True, True
        if s not in self.Ps:
            gfa_representation = self.game.gfa_to_tensor(gfa)
            self.Ps[s], v = self.nnet.predict(gfa_representation)
            v = -v.item() - 1
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
            n = self.game.n
            k = self.game.k
            d = self.game.d
            worst_case = (4 ** n) * k * d
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    self.Nsa[(s, a)] = 0
                    self.Qsa[(s, a)] = -math.log(worst_case)
            #print("v:", v)
            return v, False, False
        valids = self.Vs[s]
        cur_best = -float('inf')
        action_space = [self.Qsa[(s, a)] for a in range(self.game.getActionSize()) if (s, a) in self.Qsa]
        q_max = max(action_space)
        q_min = min(action_space)
        best_act = -1
        for a in range(self.game.getActionSize()):
            if valids[a] and (s, a) not in self.dead_end:
                exploitation = self.normalize(self.Qsa[(s, a)], q_max, q_min)
                exploration = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + self.Nsa[(s, a)])
                u = exploitation + exploration
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        if a == -1:
            return q_min, True, True
        next_gfa = self.game.getNextState(gfa, a, duplicate=True)
        v, dead_end, actual_reward = self.search(next_gfa)
        if dead_end:
            self.dead_end.add((s, a))
        if actual_reward and (s, a) not in self.actual_reward:
            self.Qsa[(s, a)] = v
            self.actual_reward.add((s, a))
        elif not actual_reward and (s, a) in self.actual_reward:
            pass
        else:
            self.Qsa[(s, a)] = max(self.Qsa[(s, a)], v)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1
        return v, False, actual_reward
