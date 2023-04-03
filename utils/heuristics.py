import copy
from queue import PriorityQueue
from random import shuffle

import networkx as nx
from FAdo.conversions import *
from FAdo.reex import CConcat, CStar, CDisj

from utils.fadomata import *

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


def decompose(gfa: GFA, state_weight: bool = False, repeated: bool = False) -> RegExp:
    final_result = None
    subautomata = decompose_vertically(gfa)
    for subautomaton in subautomata:
        result = decompose_horizontally(subautomaton, state_weight, repeated)
        final_result = result if final_result == None else CConcat(
            final_result, result)
    return final_result


def decompose_vertically(gfa: GFA) -> list:
    subautomata = []
    bridge_states = list(get_bridge_states(gfa))
    if not bridge_states:
        return [gfa]
    initial_state = gfa.Initial
    for i in range(len(bridge_states) + 1):
        final_state = bridge_states[i] if i < len(
            bridge_states) else list(gfa.Final)[0]
        subautomata.append(make_vertical_subautomaton(
            gfa, initial_state, final_state))
        initial_state = final_state
    return subautomata


def make_vertical_subautomaton(gfa: GFA, initial_state: int, final_state: int) -> GFA:
    reachable_states = list()
    check_all_reachable_states(
        gfa, initial_state, final_state, reachable_states)
    return make_subautomaton(gfa, reachable_states, initial_state, final_state)


def check_all_reachable_states(gfa: GFA, state: int, final_state: int, reachable_states: list):
    if state not in reachable_states:
        reachable_states.append(state)
        if state == final_state:
            return
        if state in gfa.delta:
            for dest in gfa.delta[state]:
                check_all_reachable_states(
                    gfa, dest, final_state, reachable_states)


def make_subautomaton(gfa: GFA, reachable_states: list, initial_state: int, final_state: int) -> GFA:
    new = GFA()
    new.States = [str(i) for i in range(len(reachable_states))]
    new.Sigma = copy.copy(gfa.Sigma)
    new.setInitial(0)
    new.setFinal([len(reachable_states) - 1])
    new.predecessors = {}
    for i in range(len(new.States)):
        new.predecessors[i] = set([])
    matching_states = {0: initial_state, len(
        reachable_states) - 1: final_state}
    counter = 1
    for i in reachable_states:
        if i not in [initial_state, final_state]:
            matching_states[counter] = i
            counter += 1
    for i in range(len(reachable_states)):
        for j in range(len(reachable_states)):
            original_state_index_1 = matching_states[i]
            original_state_index_2 = matching_states[j]
            if i is list(new.Final)[0]:
                continue
            if original_state_index_2 in gfa.delta[original_state_index_1]:
                add_transition(
                    new, i, gfa.delta[original_state_index_1][original_state_index_2], j)
    for i in range(len(new.States)):
        if i not in new.delta:
            new.delta[i] = {}
    return new


def decompose_horizontally(gfa: GFA, state_weight: bool, repeated: bool) -> RegExp:
    subautomata = []
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
    final_result = None
    for subautomaton in subautomata:
        if len(get_bridge_states(subautomaton)):
            result = decompose(subautomaton, state_weight, repeated)
        elif state_weight and repeated:
            result = eliminate_by_repeated_state_weight_heuristic(subautomaton)
        elif state_weight:
            result = eliminate_by_state_weight_heuristic(subautomaton)
        else:
            result = eliminate_randomly(subautomaton)
        final_result = result if final_result == None else CDisj(
            final_result, result)
    return final_result


def eliminate_randomly(gfa: GFA) -> RegExp:
    random_order = [i for i in range(1, len(gfa.States) - 1)]
    shuffle(random_order)
    for i in random_order:
        gfa.eliminate(i)
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def eliminate_by_state_weight_heuristic(gfa: GFA) -> RegExp:
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((get_weight(gfa, i), i))
    while not pq.empty():
        gfa.eliminate(pq.get()[1])
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def eliminate_by_repeated_state_weight_heuristic(gfa: GFA) -> RegExp:
    n = len(gfa.States) - 2
    victim = [i + 1 for i in range(len(gfa.States) - 2)]
    for i in range(n):
        if (len(victim) == 1):
            gfa.eliminate(victim[0])
            continue
        min_val = get_weight(gfa, victim[0])
        min_idx = 0
        for j in range(1, len(victim)):
            curr_val = get_weight(gfa, victim[j])
            if min_val > curr_val:
                min_val = curr_val
                min_idx = j
        gfa.eliminate(victim[min_idx])
        del victim[min_idx]
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]
