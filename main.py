from FAdo.fa import *
from FAdo.reex import *
from FAdo.fio import *
from FAdo.conversions import *

from queue import PriorityQueue


def state_weight(gfa: GFA):
    pq = PriorityQueue()
    for i in range(1, len(gfa.States) - 1):
        pq.put((gfa.weight(i), i))

    lst = list()
    while not pq.empty():
        lst.append(pq.get()[1])

    gfa.eliminateAll(lst)
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
print(state_weight_result)