from FAdo.fa import *
from FAdo.reex import *
from FAdo.conversions import *

from queue import PriorityQueue
import random

from sample import *

def random_elimination(nfa: NFA) -> RegExp:
    gfa = FA2GFA(nfa)
    random_order = [i for i in range(1, len(nfa.States) - 1)]
    #fixed seed for debuggin purpose
    random.seed(0)
    random.shuffle(random_order)
    for i in random_order:
        gfa.eliminate(i)
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def test_random_elimination():
    fig10 = get_fig10()
    print(random_elimination(fig10))

def state_weight(nfa: NFA) -> RegExp:
    gfa = FA2GFA(nfa)
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((gfa.weight(i), i))
    while not pq.empty():
        gfa.eliminate(pq.get()[1])
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def repeated_state_weight(nfa: NFA) -> RegExp:
    gfa = FA2GFA(nfa)
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
    fig10 = get_fig10()
    state_weight_result = state_weight(fig10)
    repeated_state_weight_result = repeated_state_weight(fig10)
    print(state_weight_result)
    print(repeated_state_weight_result)