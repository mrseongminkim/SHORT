from queue import PriorityQueue
from random import shuffle

from FAdo.conversions import *
from FAdo.reex import *

from fadomata import *

def decompose(gfa: GFA, state_weight: bool = False, repeated: bool = False) -> RegExp:
    final_result = None
    subautomata = decompose_vertically(gfa)
    for subautomaton in subautomata:
        result = decompose_horizontally(subautomaton, state_weight, repeated)
        final_result = result if final_result == None else CConcat(final_result, result)
    return final_result

def decompose_vertically(gfa: GFA) -> list:
    subautomata = []
    bridge_states = list(get_bridge_states(gfa))
    if not bridge_states:
        return [gfa]
    initial_state = gfa.Initial
    for i in range(len(bridge_states) + 1):
        final_state = bridge_states[i] if i < len(bridge_states) else list(gfa.Final)[0]
        subautomata.append(make_vertical_subautomaton(gfa, initial_state, final_state))
        initial_state = final_state
    return subautomata

def make_vertical_subautomaton(gfa: GFA, initial_state: int, final_state: int) -> GFA:
    reachable_states = list()
    check_all_reachable_states(gfa, initial_state, final_state, reachable_states)
    return make_subautomaton(gfa, reachable_states)

def check_all_reachable_states(gfa: GFA, state: int, final_state: int, reachable_states: list):
    if state not in reachable_states:
        reachable_states.append(state)
        if state == final_state:
            return
        if state in gfa.delta:
            for dest in gfa.delta[state]:
                check_all_reachable_states(gfa, dest, final_state, reachable_states)

def make_subautomaton(gfa: GFA, reachable_states: list) -> GFA:
        reachable_states.sort()
        new = GFA()
        new.States = [str(i) for i in range(len(reachable_states))]
        new.Sigma = copy(gfa.Sigma)
        new.setInitial(0)
        new.setFinal([len(reachable_states) - 1])
        new.predecessors = {}
        for i in range(len(new.States)):
            new.predecessors[i] = set([])
        for i in range(len(reachable_states)):
            for j in range(len(reachable_states)):
                original_state_index_1 = reachable_states[i]
                original_state_index_2 = reachable_states[j]
                copied_state_index_1 = i
                copied_state_index_2 = j
                if copied_state_index_1 is list(new.Final)[0]:
                    continue
                if original_state_index_2 in gfa.delta[original_state_index_1]:
                    new.addTransition(copied_state_index_1, gfa.delta[original_state_index_1][original_state_index_2], copied_state_index_2)
        return new

def decompose_horizontally(gfa: GFA, state_weight: bool, repeated: bool) -> RegExp:
    subautomata = []
    groups = []
    for state in gfa.delta[gfa.Initial]:
        reachable_states = [gfa.Initial]
        check_all_reachable_states(gfa, state, list(gfa.Final)[0], reachable_states)
        is_disjoint = True
        for group in groups:
            if [state for state in reachable_states if state in group] != [gfa.Initial, list(gfa.Final)[0]]:
                is_disjoint = False
                group = group + [state for state in reachable_states if state not in group]
                break
        if is_disjoint:
            groups.append(reachable_states)
    if len(groups) == 1:
        subautomata.append(gfa)
    else:
        for group in groups:
            subautomata.append(make_subautomaton(gfa, group))
    final_result = None
    for subautomaton in subautomata:
        if len(get_bridge_states(subautomaton)):
            result = decompose(subautomaton, state_weight, repeated)
        elif state_weight and repeated:
            result = repeated_state_weight_elimination(subautomaton)
        elif state_weight:
            result = state_weight_elimination(subautomaton)
        else:
            result = random_elimination(subautomaton)
        final_result = result if final_result == None else CDisj(final_result, result)
    return final_result

def random_elimination(gfa: GFA) -> RegExp:
    random_order = [i for i in range(1, len(gfa.States) - 1)]
    shuffle(random_order)
    for i in random_order:
        gfa.eliminate(i)
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def state_weight_elimination(gfa: GFA) -> RegExp:
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((get_weight(gfa, i), i))
    while not pq.empty():
        gfa.eliminate(pq.get()[1])
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def repeated_state_weight_elimination(gfa: GFA) -> RegExp:
    for i in range(len(gfa.States) - 2):
        min_val = get_weight(gfa, 1)
        min_idx = 1
        for j in range(2, len(gfa.States) - 1):
            curr_val = get_weight(gfa, j)
            if min_val > curr_val:
                min_val = curr_val
                min_idx = j
        gfa.eliminateState(min_idx)
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]