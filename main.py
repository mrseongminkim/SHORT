from FAdo.fa import *
from FAdo.reex import *
from FAdo.fio import *
from FAdo.conversions import *

from heuristics import *
from sample import *

def decomposition(nfa: NFA) -> RegExp:
    result = RegExp(nfa.Sigma)
    subautomatas = vertical_decomposition(nfa)
    for subautomata in subautomatas:
        result = CConcat(result, horizontal_decomposition(subautomata))
    return result

def vertical_decomposition(nfa: NFA) -> list:
    subautomatas = list()
    bridge_state = list(cutPoints(nfa))
    if not bridge_state:
        return [nfa]
    initial_state = list(nfa.Initial)[0]
    for i in range(len(bridge_state) + 1):
        final_state = bridge_state[i] if i < len(bridge_state) else list(nfa.Final)[0]
        subautomatas.append(get_subautomata(nfa, initial_state, final_state))
        initial_state = final_state
    return subautomatas

def get_subautomata(nfa: NFA, initial_state, final_state) -> NFA:
    new = NFA()
    new.setSigma(nfa.Sigma)
    new.States = nfa.States[initial_state : final_state + 1]
    new.Initial = {initial_state, }
    new.Final = {final_state, }
    for s in nfa.delta and new.States:
        new.delta[s] = {}
        for c in nfa.delta[s] and new.States:
            new.delta[s][c] = nfa.delta[s][c].copy()
    del new.delta[final_state]
    return new

def get_family_group(nfa: NFA) -> dict:
    count = 0
    possible_group = []
    ancester = {}
    stack = []
    visited = []
    for forefather in nfa.delta[list(nfa.Initial)]:
        stack.append(forefather)
        possible_group.append(count)
        ancester[forefather] = count
        count + 1
    while not stack.empty():
        curr = stack.pop()
        for child in nfa.delta[curr]:
            if child in ancester and ancester[child] != ancester[child]:
                pass

def horizontal_decomposition(nfa: NFA, state_weight = False, repeated = False) -> RegExp:
    family_group = get_family_group(nfa)
    for family in family_group:
        pass

    if len(cutPoints(nfa)):
        return decomposition(nfa)
    if state_weight:
        return repeated_state_weight(nfa) if repeated else state_weight(nfa)
    #return random_elimination(nfa)