from FAdo.fa import *
from FAdo.reex import *
from FAdo.fio import *
from FAdo.conversions import *

from queue import PriorityQueue

def state_weight(gfa: GFA) -> str:
    gfa = gfa.dup()
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((gfa.weight(i), i))
    while not pq.empty():
        gfa.eliminate(pq.get()[1])
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def repeated_state_weight(gfa: GFA) -> str:
    gfa = gfa.dup()
    for i in range(len(gfa.States) - 2):
        min_val = gfa.weight(1)
        min_idx = 1
        for j in range(2, len(gfa.States) - 1):
            curr_val = gfa.weight(j)
            if min_val > curr_val:
                min_val = curr_val
                min_idx = j
        gfa.eliminateState(min_idx)
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def test_state_weight():
    fig10 = FA2GFA(get_fig10())
    state_weight_result = state_weight(fig10)
    repeated_state_weight_result = repeated_state_weight(fig10)
    print(state_weight_result)
    print(repeated_state_weight_result)

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

def get_family_group(nfa: NFA) -> tuple:
    pass

def horizontal_decomposition(nfa: NFA, state_weight = False, repeated = False) -> RegExp:
    ancestor, family = get_family_group(nfa)



    if len(cutPoints(nfa)):
        return decomposition(nfa)
    if state_weight:
        return repeated_state_weight(nfa) if repeated else state_weight(nfa)
    #return random_elimination(nfa)



def get_fig10():
    fig10 = NFA()
    fig10.setSigma(['a', 'b', 'c'])
    fig10.addState('0')
    fig10.addState('1')
    fig10.addState('2')
    fig10.addState('3')
    fig10.addState('4')
    fig10.setInitial([0])
    fig10.setFinal([4])
    fig10.addTransition(0, 'a', 1)
    fig10.addTransition(1, 'b', 3)
    fig10.addTransition(2, 'b', 1)
    fig10.addTransition(2, 'a', 2)
    fig10.addTransition(2, 'b', 2)
    fig10.addTransition(2, 'c', 2)
    fig10.addTransition(2, 'b', 3)
    fig10.addTransition(3, 'c', 2)
    fig10.addTransition(3, 'a', 4)
    return fig10

def get_fig8():
    fig8 = NFA()
    fig8.setSigma(['a', 'b'])
    fig8.addState('0')
    fig8.addState('1')
    fig8.addState('2')
    fig8.addState('3')
    fig8.addState('4')
    fig8.addState('5')
    fig8.addState('6')
    fig8.setInitial([0])
    fig8.setFinal([6])
    fig8.addTransition(0, 'b', 1)
    fig8.addTransition(0, 'b', 2)
    fig8.addTransition(1, 'b', 5)
    fig8.addTransition(2, 'a', 3)
    fig8.addTransition(2, 'b', 2)
    fig8.addTransition(3, 'a', 4)
    fig8.addTransition(4, 'b', 6)
    fig8.addTransition(5, 'a', 1)
    fig8.addTransition(5, 'a', 6)
    return fig8

def get_fig7():
    fig7 = NFA()
    fig7.setSigma(['a', 'b', 'c', 'd'])
    fig7.addState('0')
    fig7.addState('1')
    fig7.addState('2')
    fig7.addState('3')
    fig7.addState('4')
    fig7.setInitial([0])
    fig7.setFinal([4])
    fig7.addTransition(0, 'c', 1)
    fig7.addTransition(1, 'a', 2)
    fig7.addTransition(2, 'b', 2)
    fig7.addTransition(2, 'd', 1)
    fig7.addTransition(2, 'a', 3)
    fig7.addTransition(3, 'b', 4)
    return fig7






#fig8 = get_fig8()
#print(fig8.evalNumberOfStateCycles())
#print(fig8.delta)
#print(fig8.predecessors)
#print(cutPoints(fig8))