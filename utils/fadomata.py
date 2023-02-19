import copy

from FAdo.conversions import *
from FAdo.reex import *

def get_weight(gfa: GFA, state: int) -> int:
    '''Counterpart of GFA.weight method'''
    weight = 0
    self_loop = 0
    if state in gfa.delta[state]:
        self_loop = 1
        weight += gfa.delta[state][state].treeLength() * (len(gfa.predecessors[state]) - self_loop) * (len(gfa.delta[state]) - self_loop)
    for i in gfa.predecessors[state]:
        if i != state and i in gfa.delta:
            weight += gfa.delta[i][state].treeLength() * (len(gfa.delta[state]) - self_loop)
    for i in gfa.delta[state]:
        if i != state:
            weight += gfa.delta[state][i].treeLength() * (len(gfa.predecessors[state]) - self_loop)
    return weight

def convert_nfa_to_gfa(nfa: NFA) -> GFA:
    '''Counterpart of FA2GFA function'''
    gfa = GFA()
    gfa.setSigma(nfa.Sigma)
    gfa.Initial = uSet(nfa.Initial)
    gfa.States = nfa.States[:]
    gfa.setFinal(nfa.Final)
    gfa.predecessors = {}
    for i in range(len(gfa.States)):
        gfa.predecessors[i] = set([])
    for s in nfa.delta:
        for c in nfa.delta[s]:
            for s1 in nfa.delta[s][c]:
                gfa.addTransition(s, c, s1)
    for i in range(len(gfa.States)):
        if i not in gfa.delta:
            gfa.delta[i] = {}
    return gfa

def add_transition(gfa: GFA, sti1: int, sym: RegExp, sti2: int):
    '''Counterpart of GFA.addTransition'''
    if sti1 not in gfa.delta:
        gfa.delta[sti1] = {}
    if sti2 not in gfa.delta[sti1]:
        gfa.delta[sti1][sti2] = sym
    else:
        gfa.delta[sti1][sti2] = reex.CDisj(gfa.delta[sti1][sti2], sym, copy.copy(gfa.Sigma))
    try:
        gfa.predecessors[sti2].add(sti1)
    except KeyError:
        pass