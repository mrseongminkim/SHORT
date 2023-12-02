from math import log

import torch
from FAdo.reex import *
from FAdo.conversions import GFA
from torch_geometric.data import Data

from SHORT.utils.random_nfa_generator import generate
from SHORT.utils.heuristics import eliminate_with_minimization
from SHORT.utils.fadomata import reverse_gfa

class StateEliminationGame():
    def __init__(self, maxN=5):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if not n or not k or not d:
            n, k, d = 5, 5, 0.1
        gfa = generate(n, k, d)
        self.n, self.k, self.d = n, k, d
        return gfa

    def get_one_hot_vector(self, state_number):
        one_hot_vector = [0] * (self.maxN + 3) #init, final, non-existing node
        one_hot_vector[state_number] = 1
        return one_hot_vector

    def gfa_to_tensor(self, gfa: GFA):
        rev = reverse_gfa(gfa)
        forward_graph = self.gfa_to_graph(gfa)
        backward_graph = self.gfa_to_graph(rev)
        return (forward_graph, backward_graph)

    #forward_only
    def gfa_to_graph(self, gfa: GFA):
        num_nodes = self.getActionSize() #7 - 5 original node and initial and final
        x = []
        edge_index = [[], []]
        for source in range(num_nodes):
            if source < len(gfa.States):
                source_state_number = self.get_one_hot_vector(int(gfa.States[source]))
                is_initial_state = 1 if source == gfa.Initial else 0
                is_final_state = 1 if source in gfa.Final else 0
                delta = []
                for target in range(num_nodes):
                    if target in gfa.delta[source]:
                        edge_index[0].append(source)
                        edge_index[1].append(target)
                        delta += self.get_one_hot_vector(int(gfa.States[target])) + [gfa.delta[source][target].treeLength()]
                    else:
                        delta += self.get_one_hot_vector(0) + [0]
                x.append(source_state_number + [is_initial_state, is_final_state] + delta)
            else:
                source_state_number = self.get_one_hot_vector(0)
                is_initial_state = is_final_state = 0
                delta = (self.get_one_hot_vector(0) + [0]) * num_nodes
                x.append(source_state_number + [is_initial_state, is_final_state] + delta)
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        graph = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        return graph

    '''
    def gfa_to_tensor(self, gfa: GFA):
        num_nodes = self.getActionSize() #7 - 5 original node and initial and final
        forward_x = []
        forward_edge_index = [[], []]
        left_edge_attr = []
        #right_x = []
        #right_edge_index = [[], []]
        #right_edge_attr = []
        for source in range(num_nodes):
            if source < len(gfa.States):
                source_state_number = self.get_one_hot_vector(int(gfa.States[source]))
                is_initial_state = 1 if source == gfa.Initial else 0
                is_final_state = 1 if source in gfa.Final else 0
                left_x.append(source_state_number + [is_initial_state, is_final_state])
                right_x.append(source_state_number + [is_final_state, is_initial_state])
                #reverse를 하면 init은 final이 되고 final은 init이 된다.
                for target in range(len(gfa.States)):
                    if target in gfa.delta[source]:
                        target_state_number = self.get_one_hot_vector(int(gfa.States[target]))
                        left_edge_index[0].append(source)
                        left_edge_index[1].append(target)
                        #source to target (original)
                        right_edge_index[0].append(target)
                        right_edge_index[1].append(source)
                        #target to source (reversed)
                        left_edge_attr.append(gfa.delta[source][target].treeLength())
                        right_edge_attr.append(gfa.delta[source][target].treeLength())
                        #left_edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]) + source_state_number + target_state_number)
                        #right_edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]) + target_state_number + source_state_number)
            else:
                left_x.append(self.get_one_hot_vector(0) + [0, 0])
                right_x.append(self.get_one_hot_vector(0) + [0, 0])
        left_x = torch.FloatTensor(left_x)
        left_edge_index = torch.LongTensor(left_edge_index)
        left_edge_attr = torch.FloatTensor(left_edge_attr)
        left_graph = Data(x=left_x, edge_index=left_edge_index, edge_attr=left_edge_attr, num_nodes=num_nodes)
        right_x = torch.FloatTensor(right_x)
        right_edge_index = torch.LongTensor(right_edge_index)
        right_edge_attr = torch.FloatTensor(right_edge_attr)
        right_graph = Data(x=right_x, edge_index=right_edge_index, edge_attr=right_edge_attr, num_nodes=num_nodes)
        return (left_graph, right_graph)
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
            return -length
        elif len(gfa.States) == 2:
            initial_state = gfa.Initial
            final_state = list(gfa.Final)[0]
            assert initial_state not in gfa.delta[initial_state]
            assert final_state not in gfa.delta[final_state]
            length = gfa.delta[initial_state][final_state].treeLength()
            return -length
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
