import copy
from queue import PriorityQueue
from random import shuffle

import networkx as nx
from FAdo.conversions import *
from FAdo.reex import CConcat, CStar, CDisj

from utils.fadomata import *

'''
def get_bridge_states(aut: GFA) -> set:
    gfa = aut.dup()
    gfa.normalize()
    # make gfa a graph instead of a digraph
    new_edges = []
    for a in gfa.delta:
        for b in gfa.delta[a]:
            new_edges.append((a, b))
    for i in new_edges:
        if i[1] not in gfa.delta:
            gfa.delta[i[1]] = {}
        else:
            gfa.delta[i[1]][i[0]] = 'x'
    for i in new_edges:
        if i[0] not in gfa.delta[i[1]]:
            gfa.delta[i[1]][i[0]] = 'x'
    # initializations needed for cut point detection
    gfa.c = 1
    gfa.num = {}
    gfa.visited = []
    gfa.parent = {}
    gfa.low = {}
    gfa.cuts = set([])
    gfa.assignNum(gfa.Initial)
    gfa.assignLow(gfa.Initial)
    # initial state is never a cut point, so it should be removed
    gfa.cuts.remove(gfa.Initial)
    cutpoints = copy.copy(gfa.cuts) - gfa.Final
    # remove self-loops and check if the cut points are in a loop
    gfa = aut.dup()
    gfa.normalize()
    for i in gfa.delta:
        if i in gfa.delta[i]:
            del gfa.delta[i][i]
    cycles = gfa.evalNumberOfStateCycles()
    for i in cycles:
        if cycles[i] != 0 and i in cutpoints:
            cutpoints.remove(i)
    if 0 in cutpoints:
        cutpoints.remove(0)
    if len(aut.States) - 1 in cutpoints:
        cutpoints.remove(len(aut.States) - 1)
    #print(cutpoints)
    return cutpoints

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
'''
#order of bridge states matters too
def get_bridge_states(gfa: GFA) -> set:
    graph = nx.Graph()
    for source in gfa.delta:
        for target in gfa.delta[source]:
            graph.add_edge(source, target)
    bridges = nx.algorithms.bridges(graph)
    bridges_states = [i[1] for i in bridges if i[1] != 0 and i[1] != len(gfa) - 1]
    new = gfa.dup()
    for i in new.delta:
        if i in new.delta[i]:
            del new.delta[i][i]
    cycles = new.evalNumberOfStateCycles()
    for i in cycles:
        if cycles[i] != 0 and i in bridges_states:
            bridges_states.remove(i)
    dead_end = []
    for i in bridges_states:
        reachable_states = []
        check_all_reachable_states(gfa, i, list(gfa.Final)[0], reachable_states)
        if list(gfa.Final)[0] not in reachable_states:
            dead_end.append(i)
    bridges_states = [x for x in bridges_states if x not in dead_end]
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
    '''
    print("states:", gfa.States)
    print("Initial:", gfa.Initial)
    print("Final:", gfa.Final)
    print("delta:", gfa.delta)
    print("bridges:", bridge_states)
    exit()
    '''
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
    #print("init:", initial_state)
    #print("final", final_state)
    #print("rechable", reachable_states)
    del reachable_states[reachable_states.index(final_state)]
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
#done reviewing
def make_subautomaton(gfa: GFA, reachable_states: list, initial_state: int, final_state: int) -> GFA:
    new = GFA()
    new.States = [gfa.States[x] for x in reachable_states]
    new.Sigma = copy.copy(gfa.Sigma)
    new.setInitial(0)
    new.setFinal([len(reachable_states) - 1])
    new.predecessors = {}
    for i in range(len(new.States)):
        new.predecessors[i] = set([])
    #key: new delta index, value: original delta index
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


#employed disjoint set (find-union data structure)
#done reviewing
def divide_groups(gfa: GFA):
    def find_parent(parent, x):
        if parent[x] != x:
            parent[x] = find_parent(parent, parent[x])
        return parent[x]
    def union_parent(parents, x, y, candidates):
        x = find_parent(parents, x)
        y = find_parent(parents, y)
        if x in candidates and y not in candidates:
            parents[y] = x
        elif x not in candidates and y in candidates:
            parents[x] = y
        elif x < y:
            parents[y] = x
        else:
            parents[x] = y
    groups = []
    candidates = [i for i in gfa.delta[0].keys() if i != 0 and i != len(gfa) - 1]
    parents = [i for i in range(len(gfa.States))]
    for i in range(1, len(gfa.States) - 1):
        for j in gfa.delta[i]:
            union_parent(parents, i, j, candidates)
    for i in candidates:
        if i == parents[i]:
            group = [j for j in range(1, len(gfa.States) - 1) if parents[j] == i]
            group = [0] + group + [len(gfa.States) - 1]
            groups.append(list(set(group)))
    return groups


#done reviewing
def decompose_horizontally(gfa: GFA, state_weight: bool, repeated: bool, minimization: bool, bridge_state_name) -> RegExp:
    subautomata = []
    groups = divide_groups(gfa)
    if len(groups) <= 1:
        subautomata.append(gfa)
    else:
        for group in groups:
            subautomata.append(make_subautomaton(gfa, group, gfa.Initial, list(gfa.Final)[0]))
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
