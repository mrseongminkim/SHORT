from FAdo.fa import *
from FAdo.reex import *
from FAdo.conversions import *

from heuristics import *
from sample import *
from data_loader import *

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

def make_subautomaton(gfa: GFA, reachable_states: list):
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

def get_bridge_states(gfa: GFA):
    new = gfa.dup()
    new_edges = []
    for a in new.delta:
        for b in new.delta[a]:
            new_edges.append((a, b))
    for i in new_edges:
        if i[1] not in new.delta:
            new.delta[i[1]] = {}
        else:
            new.delta[i[1]][i[0]] = 'x'
    for i in new_edges:
        if i[0] not in new.delta[i[1]]:
            new.delta[i[1]][i[0]] = 'x'
    # initializations needed for cut point detection
    new.c = 1
    new.num = {}
    new.visited = []
    new.parent = {}
    new.low = {}
    new.cuts = set([])
    #Check condition 2
    new.assignNum(new.Initial)
    new.assignLow(new.Initial)
    # initial state is never a cut point, so it should be removed
    new.cuts.remove(new.Initial)
    cutpoints = copy(new.cuts) - new.Final
    # remove self-loops and check if the cut points are in a loop
    new = gfa.dup()
    for i in new.delta:
        if i in new.delta[i]:
            del new.delta[i][i]
    #Check condition 3
    cycles = new.evalNumberOfStateCycles()
    for i in cycles:
        if cycles[i] != 0 and i in cutpoints:
            cutpoints.remove(i)
    return cutpoints

nfa = FA2GFA(nfa_with_no_bridge_and_two_group())
result = decompose(nfa, False, False)
print(result)