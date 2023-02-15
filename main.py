from FAdo.fa import *
from FAdo.reex import *
from FAdo.fio import *
from FAdo.conversions import *

from heuristics import *
from sample import *

def decomposition(nfa: NFA, state_weight = False, repeated = False) -> RegExp:
    result = RegExp(nfa.Sigma)
    subautomatas = vertical_decomposition(nfa, state_weight, repeated)
    for subautomata in subautomatas:
        result = CConcat(result, horizontal_decomposition(subautomata, state_weight, repeated))
    return result

def vertical_decomposition(nfa: NFA, state_weight, repeated) -> list:
    subautomatas = []
    bridge_state = list(cutPoints(nfa))
    if not bridge_state:
        return [nfa]
    initial_state = list(nfa.Initial)[0]
    for i in range(len(bridge_state) + 1):
        final_state = bridge_state[i] if i < len(bridge_state) else list(nfa.Final)[0]
        subautomatas.append(make_vertical_subautomata(nfa, initial_state, final_state))
        initial_state = final_state
    return subautomatas

def make_vertical_subautomata(nfa: NFA, initial_state, final_state) -> NFA:
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

def horizontal_decomposition(nfa: NFA, state_weight, repeated) -> RegExp:
    result = RegExp(nfa.Sigma)
    group = identify_group(nfa)
    subautomatas = make_horizontal_subautomata(nfa, group)
    for subautomata in subautomatas:
        if len(cutPoints(subautomata)):
            result = CDisj(result, decomposition(subautomata, state_weight, repeated))
        elif (state_weight):
            result = CDisj(result, repeated_state_weight(subautomata)) if repeated else CDisj(result, state_weight(subautomata))
        else:
            result = CDisj(result, random_elimination(subautomata))
    return result

def make_horizontal_subautomata(nfa: NFA, group) -> list:
    subautomatas = []
    for sub_group in group:
        pass

def identify_group(nfa: NFA) -> list:
    gfa = FA2GFA(nfa)
    index = 0
    ancestor = {}
    visited = []
    group_index = [[i] for i in range(len(gfa.delta[gfa.Initial]))]
    for s in gfa.delta[gfa.Initial]:
        ancestor[s] = index
        index += 1
        identify_group_visit(gfa, s, ancestor, visited, group_index)
    group_index = [i for i in group_index if i]
    group = [[x for x in ancestor if ancestor[x] in group_index[i]] + [gfa.Initial, list(gfa.Final)[0]] for i in range(len(group_index))]
    return group

def identify_group_visit(gfa: GFA, s, ancestor, visited, group_index):
    if s not in visited:
        visited.append(s)
        if s in gfa.delta:
            for dest in gfa.delta[s]:
                if dest in ancestor:
                    if ancestor[s] != ancestor[dest]:
                        min_index = min(ancestor[s], ancestor[dest])
                        max_index = max(ancestor[s], ancestor[dest])
                        group_index[min_index].append(max_index)
                        group_index[max_index].pop()
                else:
                    ancestor[dest] = ancestor[s]
                identify_group_visit(gfa, s, ancestor, visited, group_index)

def test_identify_group():
    nfa = nfa_for_identify_group_test()
    print(identify_group(nfa))


def test_decomposition(state_weight = False, repeated = False):
    nfa = get_fig8()
    print(nfa.States)
    result = decomposition(nfa, state_weight, repeated)
    print(result)

test_decomposition()