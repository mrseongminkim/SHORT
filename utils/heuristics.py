import copy
from queue import PriorityQueue
from random import shuffle

import networkx as nx
from FAdo.conversions import *
from FAdo.reex import CConcat, CStar, CDisj

from utils.fadomata import *

'''
To Do
1. refactor group division
2. refactor bridge states
3. test it
'''



def get_bridge_states(gfa: GFA) -> set:
    graph = nx.Graph()
    for source in gfa.delta:
        for target in gfa.delta[source]:
            graph.add_edge(source, target)
    bridges = set(nx.algorithms.bridges(graph))
    bridges_states = {i[1] for i in bridges}
    bridges_states = bridges_states.difference(gfa.Final.union({gfa.Initial}))
    new = gfa.dup()
    for i in new.delta:
        if i in new.delta[i]:
            del new.delta[i][i]
    cycles = new.evalNumberOfStateCycles()
    for i in cycles:
        if cycles[i] != 0 and i in bridges_states:
            bridges_states.remove(i)
    for i in list(bridges_states):
        reachable_states = []
        check_all_reachable_states(
            gfa, i, list(gfa.Final)[0], reachable_states)
        if list(gfa.Final)[0] not in reachable_states:
            bridges_states.remove(i)
    return bridges_states


#done reviewing
def decompose(gfa: GFA, state_weight=False, repeated=False, minimization=False, random_order=None, bridge_state_name=None) -> RegExp:
    final_result = None
    if not bridge_state_name:
        bridge_state_name = [] #name of the states, not index of states
    subautomata = decompose_vertically(gfa, bridge_state_name)
    for subautomaton in subautomata:
        result = decompose_horizontally(subautomaton, state_weight, repeated, minimization, bridge_state_name)
        final_result = result if final_result is None else CConcat(final_result, result)
    if random_order:
        bridge_state_index = [gfa.States.index(x) for x in bridge_state_name]
        random_order = [x for x in random_order if x not in bridge_state_index]
        random_order += bridge_state_index
        final_result = eliminate_randomly(gfa, minimization, random_order)
    return final_result


#done reviewing
def decompose_vertically(gfa: GFA, bridge_state_name: list) -> list:
    subautomata = []
    bridge_states = list(get_bridge_states(gfa))
    bridge_state_name += [gfa.States[x] for x in bridge_states]
    if not bridge_states:
        return [gfa]
    initial_state = gfa.Initial
    for i in range(len(bridge_states) + 1):
        final_state = bridge_states[i] if i < len(bridge_states) else list(gfa.Final)[0]
        subautomata.append(make_vertical_subautomaton(gfa, initial_state, final_state))
        initial_state = final_state
    return subautomata


#done reviewing
def make_vertical_subautomaton(gfa: GFA, initial_state: int, final_state: int) -> GFA:
    reachable_states = list()
    check_all_reachable_states(gfa, initial_state, final_state, reachable_states)
    del reachable_states[final_state]
    reachable_states.append(final_state)
    return make_subautomaton(gfa, reachable_states, initial_state, final_state)


#check all reachable states from given state, not necessarily includes final state
#done reviewing
def check_all_reachable_states(gfa: GFA, state: int, final_state: int, reachable_states: list):
    if state not in reachable_states:
        reachable_states.append(state)
        if state == final_state:
            return
        if state in gfa.delta:
            for dest in gfa.delta[state]:
                check_all_reachable_states(gfa, dest, final_state, reachable_states)


#initial as 0, final as -1
#states name should not be modified
#Partially reviewed
def make_subautomaton(gfa: GFA, reachable_states: list, initial_state: int, final_state: int) -> GFA:
    new = GFA()
    new.States = [gfa.States[x] for x in reachable_states]
    new.Sigma = copy.copy(gfa.Sigma)
    new.setInitial(0)
    new.setFinal([len(reachable_states) - 1])
    new.predecessors = {}
    for i in range(len(new.States)):
        new.predecessors[i] = set([])
    matching_states = {0: initial_state, len(reachable_states) - 1: final_state}
    counter = 1
    for i in reachable_states[1:-1]:
        matching_states[counter] = i
        counter += 1
    for i in range(len(reachable_states)):
        for j in range(len(reachable_states)):
            original_state_index_1 = matching_states[i]
            original_state_index_2 = matching_states[j]
            if i is list(new.Final)[0]:
                continue
            if original_state_index_2 in gfa.delta[original_state_index_1]:
                add_transition(new, i, gfa.delta[original_state_index_1][original_state_index_2], j)
    for i in range(len(new.States)):
        if i not in new.delta:
            new.delta[i] = {}
    return new

'''
idea: employ disjoint set(find-union)
reviewed except group division process
'''
def decompose_horizontally(gfa: GFA, state_weight: bool, repeated: bool, minimization: bool, bridge_state_name) -> RegExp:
    subautomata = []
    #group division start
    groups = []
    for state in gfa.delta[gfa.Initial]:
        if state == list(gfa.Final)[0]:
            continue
        reachable_states = [gfa.Initial]
        check_all_reachable_states(
            gfa, state, list(gfa.Final)[0], reachable_states)
        if list(gfa.Final)[0] not in reachable_states:
            continue
        is_disjoint = True
        for group in groups:
            if [state for state in reachable_states if state in group] != [gfa.Initial, list(gfa.Final)[0]]:
                is_disjoint = False
                group = group + \
                    [state for state in reachable_states if state not in group]
                break
        if is_disjoint:
            groups.append(reachable_states)
    if len(groups) <= 1:
        subautomata.append(gfa)
    else:
        for group in groups:
            subautomata.append(make_subautomaton(
                gfa, group, gfa.Initial, list(gfa.Final)[0]))
    #group division ended
    final_result = None
    for subautomaton in subautomata:
        if len(get_bridge_states(subautomaton)):
            result = decompose(subautomaton, state_weight, repeated, minimization, bridge_state_name=bridge_state_name)
        elif state_weight and repeated:
            result = eliminate_by_repeated_state_weight_heuristic(subautomaton, minimization)
        elif state_weight:
            result = eliminate_by_state_weight_heuristic(subautomaton, minimization)
        else:
            result = None
        final_result = result if final_result == None else CDisj(final_result, result)
    return final_result


def eliminate_randomly(gfa: GFA, minimization, random_order) -> RegExp:
    for i in random_order:
        eliminate_with_minimization(gfa, i, delete_state=False, minimize=minimization)
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def eliminate_by_state_weight_heuristic(gfa: GFA, minimization) -> RegExp:
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((get_weight(gfa, i), i))
    while not pq.empty():
        eliminate_with_minimization(gfa, pq.get()[1], delete_state=False, minimize=minimization)
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def eliminate_by_repeated_state_weight_heuristic(gfa: GFA, minimization) -> RegExp:
    n = len(gfa.States) - 2
    for i in range(n):
        min_val = get_weight(gfa, 1)
        min_idx = 1
        for j in range(2, len(gfa.States) - 1):
            curr_val = get_weight(gfa, j)
            if min_val > curr_val:
                min_val = curr_val
                min_idx = j
        eliminate_with_minimization(gfa, min_idx, minimize=minimization)
    return gfa.delta[0][1]
