from FAdo.fa import *
from FAdo.reex import *
from FAdo.conversions import *

from queue import PriorityQueue
import random

from sample import *

def random_elimination(gfa: GFA) -> RegExp:
    random_order = [i for i in range(1, len(gfa.States) - 1)]
    #fixed seed for debuggin purpose
    random.seed(0)
    random.shuffle(random_order)
    for i in random_order:
        gfa.eliminate(i)
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def test_random_elimination():
    nfa = nfa_with_no_bridge_and_single_group()
    random_elimination_result = random_elimination(nfa)
    print(random_elimination_result)

def state_weight_elimination(gfa: GFA) -> RegExp:
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((gfa.weight(i), i))
    while not pq.empty():
        gfa.eliminate(pq.get()[1])
    return gfa.delta[gfa.Initial][list(gfa.Final)[0]]

def repeated_state_weight_elimination(gfa: GFA) -> RegExp:
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

def test_state_weight_elimination():
    nfa = nfa_with_no_bridge_and_single_group()
    state_weight_elimination_result = state_weight_elimination(nfa)
    repeated_state_weight_elimination_result = repeated_state_weight_elimination(nfa)
    print(state_weight_elimination_result)
    print(repeated_state_weight_elimination_result)

def bridge_states(gfa: GFA):
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