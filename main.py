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

def get_fig10():
    fig10 = NFA()
    fig10.setSigma(['a', 'b', 'c'])
    fig10.addState('0')
    fig10.addState('1')
    fig10.addState('2')
    fig10.addState('3')
    fig10.addState('4')
    fig10.setInitial([0])
    fig10.addFinal(4)
    fig10.addTransition(0, 'a', 1)
    fig10.addTransition(1, 'b', 3)
    fig10.addTransition(2, 'b', 1)
    fig10.addTransition(2, 'a', 2)
    fig10.addTransition(2, 'b', 2)
    fig10.addTransition(2, 'c', 2)
    fig10.addTransition(2, 'b', 3)
    fig10.addTransition(3, 'c', 2)
    fig10.addTransition(3, 'a', 4)
    return FA2GFA(fig10)

fig10 = get_fig10()
state_weight_result = state_weight(fig10)
repeated_state_weight_result = repeated_state_weight(fig10)
print(state_weight_result)
print(repeated_state_weight_result)