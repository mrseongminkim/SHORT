from math import log

import torch
from FAdo.reex import *
from FAdo.conversions import GFA
from torch_geometric.data import Data

from utils.random_nfa_generator import generate
from utils.heuristics import eliminate_with_minimization
from utils.fadomata import reverse_gfa

from config import *

#0 as emptyness, 5 + max(ALPHABET) should be input size of embedding dimension
word_to_ix = {'@': 1, '+': 2, '*': 3, '.': 4}
for i in range(max(ALPHABET)):
    word_to_ix[str(i)] = i + 5

class StateEliminationGame():
    def __init__(self, maxN=MAX_STATES):
        self.maxN = maxN

    def get_initial_gfa(self, gfa=None, n=None, k=None, d=None):
        if not n or not k or not d:
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
        one_hot_vector = [0] * (self.maxN + 3) #init, final, non-existing node
        one_hot_vector[state_number] = 1
        return one_hot_vector

    def gfa_to_tensor(self, gfa: GFA) -> tuple[Data, Data]:
        rev = reverse_gfa(gfa)
        forward_graph = self.gfa_to_graph(gfa)
        backward_graph = self.gfa_to_graph(rev)
        return (forward_graph, backward_graph)

    #forward_only
    def gfa_to_graph(self, gfa: GFA):
        num_nodes = self.getActionSize() #52
        x = []
        edge_index = [[], []]
        edge_attr = []
        for source in range(num_nodes):
            if source < len(gfa.States):
                source_state_number = self.get_one_hot_vector(int(gfa.States[source]))
                is_initial_state = 1 if source == gfa.Initial else 0
                is_final_state = 1 if source in gfa.Final else 0
                out_connectivity = [0] * (self.maxN + 3)
                out_length = [0] * (self.maxN + 3)
                out_regex = [0] * (self.maxN + 3)
                in_connectivity = [0] * (self.maxN + 3)
                in_length = [0] * (self.maxN + 3)
                in_regex = [0] * (self.maxN + 3)
                for target in range(num_nodes):
                    if target in gfa.delta[source]:
                        edge_index[0].append(source)
                        edge_index[1].append(target)
                        target_id = int(gfa.States[target])
                        out_connectivity[target_id] = 1
                        out_length[target_id] = gfa.delta[source][target].treeLength()
                        edge_attr.append(self.get_encoded_regex(gfa.delta[source][target]))
                for predecessor in gfa.predecessors[source]:
                    predecessor_id = int(gfa.States[predecessor])
                    in_connectivity[predecessor_id] = 1
                    in_length[predecessor_id] = gfa.delta[predecessor][source].treeLength()
                out_transition = out_connectivity + out_length + out_regex
                in_transition = in_connectivity + in_length + in_regex
                x.append(source_state_number + [is_initial_state, is_final_state] + in_transition + out_transition)
            else:
                source_state_number = self.get_one_hot_vector(0)
                is_initial_state = is_final_state = 0
                out_connectivity = [0] * (self.maxN + 3)
                out_length = [0] * (self.maxN + 3)
                out_regex = [0] * (self.maxN + 3)
                in_connectivity = [0] * (self.maxN + 3)
                in_length = [0] * (self.maxN + 3)
                in_regex = [0] * (self.maxN + 3)
                out_transition = out_connectivity + out_length + out_regex
                in_transition = in_connectivity + in_length + in_regex
                x.append(source_state_number + [is_initial_state, is_final_state] + in_transition + out_transition)
        x = torch.FloatTensor(x)
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.LongTensor(edge_attr)
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)
        return graph
        
    def getActionSize(self):
        return self.maxN + 2

    def getNextState(self, gfa, action, duplicate=False, minimize=True):
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
