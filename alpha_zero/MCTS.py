#솔직히 제일 의심가기도 하면서 - 잘 작동하는 것 같기는 함
#언제나 옵티멀 경로를 뽑아내니까
#다만 리워드가 로그스케일이고 차이가 적어서 문제가 있긴 함
#의심 가는 부분도 많다.

import logging
import math

import numpy as np

from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame

from config import *

log = logging.getLogger(__name__)

class MCTS():
    def __init__(self, game, nnet):
        self.game: StateEliminationGame = game
        self.nnet = nnet
        self.Qsa = {} #음의 로그 값
        self.Nsa = {}
        self.Ns = {}
        self.Ps = {}
        self.Es = {}
        self.Vs = {}
        #error-prone
        self.dead_end = set()
        self.actual_reward = set()

    def normalize(self, q, q_max, q_min):
        #q_max, q_min 모두 음의 로그 값임
        return (q - q_min) / (q_max - q_min + EPS)

    def getActionProb(self, gfa):
        '''
        #Case: init 부터 hop이 큰 것부터 지우기
        from collections import deque
        counts = np.array([0 for _ in range(self.game.getActionSize())])
        visited = set([gfa.Initial])
        q = deque([gfa.Initial])
        while q:
            cur = q.popleft()
            for next in gfa.delta[cur]:
                if next not in visited:
                    counts[next] = counts[cur] + 1
                    q.append(next)
                    visited.add(next)
        counts[list(gfa.Final)[0]] = 0
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

        #Case: state weight가 큰 것먼저 지우게 하기
        from utils.fadomata import get_weight
        counts = np.array([0 for _ in range(self.game.getActionSize())])
        for i in range(len(gfa.States)):
            if i != gfa.Initial and i not in gfa.Final:
                counts[i] = get_weight(gfa, i)
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

        #Case: state 숫자가 큰 것 지우게 하기
        counts = np.array([0 for _ in range(self.game.getActionSize())])
        for i in range(len(gfa.States)):
            if i != gfa.Initial and i not in gfa.Final:
                counts[i] = int(gfa.States[i])
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        #print(probs)
        return probs
        '''

        for _ in range(NUMBER_OF_MCTS_SIMULATIONS):
            _, dead_end, _ = self.search(gfa)
            if dead_end:
                break
        s = self.game.stringRepresentation(gfa)
        action_space = [self.Qsa[(s, a)] for a in range(self.game.getActionSize()) if (s, a) in self.Qsa]
        q_max = max(action_space)
        q_min = min(action_space)
        if q_max == q_min:
            counts = [2 if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
        else:
            counts = [self.normalize(self.Qsa[(s, a)], q_max, q_min) + 1 if (s, a) in self.Qsa else 0 for a in range(self.game.getActionSize())]
        counts = np.array(counts)
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, gfa):
        #returns reward, is_dead_end, is_actual_reward
        s = self.game.stringRepresentation(gfa)
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(gfa)
        if self.Es[s] != None:
            #dead_end이면서 actual reward
            return self.Es[s], True, True
        if s not in self.Ps:
            #처음 방문한 노드
            gfa_representation = self.game.gfa_to_tensor(gfa)
            self.Ps[s], v = self.nnet.predict(gfa_representation)
            #그리고 이 v는 NN이 GFA의 양의 로그 스케일을 출력했다 가정
            #한 마디로 NN은 안 좋은 경로일 수록 큰 양수를 출력하게 함
            #NN should return positive value
            if v > 0:
                v = -v.item()
            else:
                #print("If this line excuted after # of training, it indicates NN is not working properly")
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
                    self.Qsa[(s, a)] = -math.log(worst_case)
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
