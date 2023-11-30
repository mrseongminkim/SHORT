import logging
import math

import numpy as np

from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from utils.fadomata import get_weight

from config import *

log = logging.getLogger(__name__)

class MCTS():
    def __init__(self, game, nnet):
        self.game: StateEliminationGame = game
        self.nnet = nnet
        self.Qsa = {} #Qsa에는 길이의 반수가 들어간다.
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        self.dead_end = set()
        self.actual_reward = set()

    #def normalize(self, q, q_max, q_min):
    #    return (q - q_min) / (q_max - q_min + EPS)

    def getActionProb(self, gfa):
        '''
        여기서 굳이 step할 필요도 없음
        state weight가 작은 것들이 높은 점수를 가지도록하면 될듯
        기본적으로 작은 거 먼저 지워야 한다.

        '''
        state_weight = [0] * self.game.getActionSize()
        min_weight = float("inf")
        min_idx = -1
        for i in range(len(gfa.States)):
            if i == gfa.Initial or i in gfa.Final:
                continue
            weight = get_weight(gfa, i)
            if weight < min_weight:
                min_weight = weight
                min_idx = i
        state_weight[min_idx] = 1
        return state_weight
        
        

        for _ in range(NUMBER_OF_MCTS_SIMULATIONS):
            _, dead_end, _ = self.search(gfa)
            if dead_end:
                break
        s = self.game.stringRepresentation(gfa)

        #print("validation:", [-self.Qsa[(s, a)] if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())])

        c = max([self.Qsa[(s, a)] for a in range(self.game.getActionSize()) if (s, a) in self.Qsa])
        denom = sum([np.e ** (self.Qsa[(s, a)] - c) for a in range(self.game.getActionSize()) if (s, a) in self.Qsa])
        numer = [np.e ** (self.Qsa[(s, a)] - c) if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
        probs = [x / denom if x != 0 else 0 for x in numer]
        return probs

    def search(self, gfa): # -> reward, is_dead_end, is_actual_reward
        s = self.game.stringRepresentation(gfa)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(gfa) #길이의 반수를 반환한다.
        if self.Es[s] != None:
            #길이의 반수, dead_end이면서 actual reward
            #self.Es[s]는 음수
            return self.Es[s], True, True
        if s not in self.Ps:
            #처음 방문한 노드
            gfa_representation = self.game.gfa_to_tensor(gfa)
            self.Ps[s], v = self.nnet.predict(gfa_representation)
            '''
            NN이 양수 길이를 반환하도록 하였기에 이를 다시 반전시킨다.
            v는 음수
            '''
            if v > 0:
                #v는 -길이
                v = -v.item()
            else:
                #print("If this line excuted after # of training, it indicates value head is not working properly")
                v = v.item()
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
            #논문 다시 보고 수정할 것
            worst_case = (4 ** n) * k * d
            for a in range(self.game.getActionSize()):
                if valids[a]:
                    self.Nsa[(s, a)] = 0
                    self.Qsa[(s, a)] = -worst_case
            return v, False, False
        valids = self.Vs[s]
        cur_best = -float('inf')
        action_space = [self.Qsa[(s, a)] for a in range(self.game.getActionSize()) if (s, a) in self.Qsa]
        #q_max = max(action_space)
        #q_min = min(action_space)
        best_act = -1
        for a in range(self.game.getActionSize()):
            if valids[a] and (s, a) not in self.dead_end:
                exploitation = self.Qsa[(s, a)]
                #exploitation = self.normalize(self.Qsa[(s, a)], q_max, q_min)
                exploration = CPUCT * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS) / (1 + self.Nsa[(s, a)])
                u = exploitation + exploration
                if u > cur_best:
                    cur_best = u
                    best_act = a
        a = best_act
        if a == -1:
            #모든 action이 valid하지 않거나(일어나면 안 된다) or 이미 모든 하위 노드들을 방문했을 때
            #현재 시점에서 달성할 수 있는 최고의 값을 리포트해야한다.
            #다 음수값이니 맥스하면 된다.
            #여기서 최종적으로 반환이 된다.
            return max(action_space), True, True
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
