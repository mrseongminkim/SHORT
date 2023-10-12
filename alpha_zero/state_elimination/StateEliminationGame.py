from math import log
from collections import defaultdict

import torch
from FAdo.reex import *

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

    def get_encoded_regex(self, regex=None):
        if regex == None:
            return [0] * MAX_LEN
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

    def get_transitions(self, gfa):
        in_transitions = defaultdict(set)
        out_transitions = defaultdict(set)
        for i in gfa.delta:
            for j in gfa.delta[i]:
                in_transitions[j].add(i)
                out_transitions[i].add(j)
        return in_transitions, out_transitions

    def gfa_to_tensor(self, gfa):
        action_size = self.getActionSize()
        in_transitions, out_transitions = self.get_transitions(gfa)
        tensor = []
        for i in range(action_size):
            if i < len(gfa.States):
                source_state_number = self.get_one_hot_vector(int(gfa.States[i]))
            else:
                source_state_number = self.get_one_hot_vector(0)
            is_initial_state = [1] if i == gfa.Initial else [0]
            is_final_state = [1] if i in gfa.Final else [0]
            in_transition = []
            in_transition_state = []
            out_transition = []
            out_transition_state = []
            for j in range(action_size):
                if j == len(gfa.States):
                    in_transition += self.get_encoded_regex() * (action_size - len(gfa.States))
                    out_transition += self.get_encoded_regex() * (action_size - len(gfa.States))
                    in_transition_state += self.get_one_hot_vector(0) * (action_size - len(gfa.States))
                    out_transition_state += self.get_one_hot_vector(0) * (action_size - len(gfa.States))
                    break
                #이거 그냥 delta 쓰면 되잖아;;;
                if j in in_transitions[i]:
                    in_transition += self.get_encoded_regex(gfa.delta[j][i])
                    in_transition_state += self.get_one_hot_vector(int(gfa.States[j]))
                else:
                    in_transition += self.get_encoded_regex()
                    in_transition_state += self.get_one_hot_vector(0)
                if j in out_transitions[i]:
                    out_transition += self.get_encoded_regex(gfa.delta[i][j])
                    out_transition_state += self.get_one_hot_vector(int(gfa.States[j]))
                else:
                    out_transition += self.get_encoded_regex()
                    out_transition_state += self.get_one_hot_vector(0)
            state_information = source_state_number + is_initial_state + is_final_state + in_transition + in_transition_state + out_transition + out_transition_state
            tensor.append(state_information)
        return tensor

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
        validMoves = [0 for _ in range(self.maxN + 2)]
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
        if len(gfa.States) == 3:
            result = self.get_resulting_regex(gfa)
            length = result.treeLength()
            reward = -log(length)
            return reward
        elif len(gfa.States) == 2:
            initial_state = gfa.Initial
            final_state = list(gfa.Final)[0]
            assert initial_state not in gfa.delta[initial_state]
            assert final_state not in gfa.delta[final_state]
            length = gfa.delta[initial_state][final_state].treeLength()
            reward = -log(length)
            return reward
        else:
            return None

    def stringRepresentation(self, gfa):
        return str(gfa.delta)
