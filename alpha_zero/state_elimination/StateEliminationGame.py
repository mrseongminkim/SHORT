from math import log

import torch
from FAdo.reex import *
from torch_geometric.data import Data

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization

from config import *

#0 as emptyness, 5 + max(ALPHABET) should be input size of embedding dimension
word_to_ix = {'@': 1, '+': 2, '*': 3, '.': 4}
for i in range(max(ALPHABET)):
    word_to_ix[str(i)] = i + 5

class StateEliminationGame():
    def __init__(self, maxN=MAX_STATES):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if gfa is None:
            n, k, d = 5, 5, 0.1
            gfa = generate(n, k, d)
        self.n, self.k, self.d = n, k, d
        return gfa

    def get_encoded_regex(self, regex):
        #NB: This technique cannot be applied to GFA with an alphabet size of more than 9.
        encoded_regex = [word_to_ix[word] for word in list(regex.rpn().replace("@epsilon", "@").replace("'", ""))[:MAX_LEN]]
        if len(encoded_regex) < MAX_LEN:
            encoded_regex = encoded_regex + [0] * (MAX_LEN - len(encoded_regex))
        assert len(encoded_regex) == MAX_LEN
        return encoded_regex
    
    def get_one_hot_vector(self, state_number):
        one_hot_vector = [0] * (self.maxN + 3)
        one_hot_vector[state_number] = 1
        return one_hot_vector

    def gfa_to_tensor(self, gfa):
        num_nodes = self.getActionSize()
        x = []
        edge_index = [[], []]
        edge_attr = []
        for source in range(num_nodes):
            if source < len(gfa.States):
                source_state_number = self.get_one_hot_vector(int(gfa.States[source]))
                is_initial_state = 1 if source == gfa.Initial else 0
                is_final_state = 1 if source in gfa.Final else 0
                x.append(source_state_number + [is_initial_state, is_final_state])
                for target in range(len(gfa.States)):
                    if target in gfa.delta[source]:
                        target_state_number = self.get_one_hot_vector(int(gfa.States[target]))
                        edge_index[0].append(source)
                        edge_index[1].append(target)
                        edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]) + source_state_number + target_state_number)
            else:
                x.append(self.get_one_hot_vector(0) + [0, 0])
                #노드 정보만 있고 연결성은 없앤다...
                #사실 그게 맞지
                #self-connection을 달아줬던 이유는 단순히 기존 모델과 충돌이 안 나게 하려고 한 것이니까.
                #edge_index[0].append(source)
                #edge_index[1].append(source)
                #edge_attr.append([0] * MAX_LEN + self.get_one_hot_vector(0) + self.get_one_hot_vector(0))
        x = torch.LongTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.LongTensor(edge_attr)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        return graph

    '''
    def gfa_to_tensor(self, gfa):
        num_nodes = len(gfa.States)
        x = []
        edge_index = [[], []]
        edge_attr = []
        for source in range(len(gfa.States)):
            source_state_number = self.get_one_hot_vector(int(gfa.States[source]))
            is_initial_state = 1 if source == gfa.Initial else 0
            is_final_state = 1 if source in gfa.Final else 0
            x.append(source_state_number + [is_initial_state, is_final_state])
            for target in range(len(gfa.States)):
                if target in gfa.delta[source]:
                    target_state_number = self.get_one_hot_vector(int(gfa.States[target]))
                    edge_index[0].append(source)
                    edge_index[1].append(target)
                    edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]) + source_state_number + target_state_number)
        x = torch.LongTensor(x)
        assert num_nodes == len(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.LongTensor(edge_attr)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        return graph
    '''

    def getActionSize(self):
        return self.maxN + 2

    def getNextState(self, gfa, action, duplicate=False, minimize=False):
        initial_state = gfa.Initial
        final_state = list(gfa.Final)[0]
        assert action < len(gfa.States) and action != final_state and action != initial_state
        if duplicate:
            eliminated_gfa = eliminate_with_minimization(gfa.dup(), action, minimize=minimize)
        else:
            eliminated_gfa = eliminate_with_minimization(gfa, action, minimize=minimize)
        return eliminated_gfa

    def getValidMoves(self, gfa):
        initial_state = gfa.Initial
        final_state = list(gfa.Final)[0]
        validMoves = [0 for i in range(self.maxN + 2)]
        for i in range(len(gfa.States)):
            if i != initial_state and i != final_state:
                validMoves[i] = 1
        return validMoves
    
    def get_resulting_regex(self, gfa):
        initial_state = gfa.Initial
        final_state = list(gfa.Final)[0]
        intermediate_state = 3 - (initial_state + final_state)

        alpha = gfa.delta[initial_state][intermediate_state]
        beta = CStar(gfa.delta[intermediate_state][intermediate_state]) if intermediate_state in gfa.delta[intermediate_state] else None
        gamma = gfa.delta[intermediate_state][final_state]
        direct_edge = gfa.delta[initial_state][final_state] if final_state in gfa.delta[initial_state] else None

        result = CConcat(CConcat(alpha, beta), gamma) if beta is not None else CConcat(alpha, gamma)
        result = CDisj(direct_edge, result) if direct_edge is not None else result

        return result

    def getGameEnded(self, gfa):
        #state가 3이면 init-bridge-final의 linear한 형상을 가짐
        #init -> final의 direct edge와 linear한 edge를 이어주면 됨
        if len(gfa.States) == 3:
            result = self.get_resulting_regex(gfa)
            length = result.treeLength()
            #로그를 취해서 값이 너무 작아짐 -> 작은 차이가 없어지는 것이 문제
            reward = -log(length)
            return reward
        #n=3인 GFA에서 간혹 나올 수 있어서 예외로 처리해줌
        elif len(gfa.States) == 2:
            initial_state = gfa.Initial
            final_state = list(gfa.Final)[0]
            assert initial_state not in gfa.delta[initial_state]
            assert final_state not in gfa.delta[final_state]
            length = gfa.delta[initial_state][final_state].treeLength()
            reward = -log(length)
            #return 문이 없었는데... 어차피 이 상황은 test에나 발생해서 문제 없었다.
            return reward
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
