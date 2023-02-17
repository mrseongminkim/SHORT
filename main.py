from FAdo.fa import *
from FAdo.reex import *
from FAdo.fio import *
from FAdo.conversions import *

from heuristics import *
from sample import *
from data_loader import *

def decomposition(nfa: NFA, state_weight = False, repeated = False) -> RegExp:
    result = RegExp(nfa.Sigma)
    subautomatas = vertical_decomposition(nfa, state_weight, repeated)
    for subautomata in subautomatas:
        result = CConcat(result, horizontal_decomposition(subautomata, state_weight, repeated))
    return result

def vertical_decomposition(nfa: NFA, state_weight, repeated) -> list:
    subautomatas = []
    bridge_state = list(cutPoints(nfa))
    if not bridge_state:
        return [nfa]
    initial_state = list(nfa.Initial)[0]
    for i in range(len(bridge_state) + 1):
        final_state = bridge_state[i] if i < len(bridge_state) else list(nfa.Final)[0]
        subautomatas.append(make_vertical_subautomata(nfa, initial_state, final_state))
        initial_state = final_state
    return subautomatas

def make_vertical_subautomata(nfa: NFA, initial_state, final_state) -> NFA:
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





















def identify_group(gfa: GFA) -> list:
    index = 0
    ancestor = {}
    visited = []
    group_index = [[i] for i in range(len(gfa.delta[gfa.Initial]))]
    for s in gfa.delta[gfa.Initial]:
        if s not in ancestor:
            ancestor[s] = index
        else:
            group_index[ancestor[s]].append(index)
            group_index[index].pop()
        index += 1
        identify_group_visit(gfa, s, ancestor, visited, group_index)
    group_index = [i for i in group_index if i]
    group = [[x for x in ancestor if ancestor[x] in group_index[i]] + [gfa.Initial, list(gfa.Final)[0]] for i in range(len(group_index))]
    return group

def identify_group_visit(gfa: GFA, s, ancestor, visited, group_index):
    if s not in visited:
        visited.append(s)
        if s in gfa.delta:
            for dest in gfa.delta[s]:
                if dest == list(gfa.Final)[0]:
                    continue
                if dest in ancestor:
                    if ancestor[s] != ancestor[dest]:
                        min_index = min(ancestor[s], ancestor[dest])
                        max_index = max(ancestor[s], ancestor[dest])
                        group_index[min_index].append(max_index)
                        group_index[max_index].pop()
                else:
                    ancestor[dest] = ancestor[s]
                identify_group_visit(gfa, dest, ancestor, visited, group_index)

def test_identify_group():
    nfa = nfa_for_identify_group_test()
    print(identify_group(nfa))

def test_decomposition(state_weight = False, repeated = False):
    nfa = get_fig8()
    print(nfa.States)
    result = decomposition(nfa, state_weight, repeated)
    print(result)



'''
    def dup(self):
        """ Returns a copy of a GFA

        :rtype: GFA"""
        new = GFA()
        new.States = copy(self.States)
        new.Sigma = copy(self.Sigma)
        new.Initial = self.Initial
        new.Final = copy(self.Final)
        new.delta = deepcopy(self.delta)
        new.predecessors = deepcopy(self.predecessors)
        return new

    def addTransition(self, sti1, sym, sti2):
        """Adds a new transition from ``sti1`` to ``sti2`` consuming symbol ``sym``. Label of the transition function
         is a RegExp.

        :param int sti1: state index of departure
        :param int sti2: state index of arrival
        :param str sym: symbol consumed
        :raises DFAepsilonRedefenition: if sym is Epsilon"""
        try:
            self.addSigma(sym)
            sym = reex.CAtom(sym, copy(self.Sigma))
        except DFAepsilonRedefinition:
            sym = reex.CEpsilon(copy(self.Sigma))
        if sti1 not in self.delta:
            self.delta[sti1] = {}
        if sti2 not in self.delta[sti1]:
            self.delta[sti1][sti2] = sym
        else:
            self.delta[sti1][sti2] = reex.CDisj(self.delta[sti1][sti2], sym, copy(self.Sigma))
        # TODO: write cleaner code and get rid of the general catch
        # noinspection PyBroadException
        try:
            self.predecessors[sti2].add(sti1)
        except KeyError:
            pass
'''

def make_horizontal_subautomata(gfa: GFA, group) -> list:
    if len(group) == 1:
        return [gfa]
    subautomata = []
    for index in group:
        index.sort()
        new = GFA()
        new.States = [str(i) for i in range(len(index))]
        new.Sigma = copy(gfa.Sigma)
        new.setInitial(0)
        new.setFinal([len(index) - 1])
        new.predecessors = {}
        for i in range(len(new.States)):
            new.predecessors[i] = set([])
        #could be solve with DFS more efficiently, let's optimize it later
        for i in range(len(index)):
            for j in range(len(index)):
                original_state_index_1 = index[i]
                original_state_index_2 = index[j]
                copied_state_index_1 = i
                copied_state_index_2 = j
                if original_state_index_1 is list(gfa.Final)[0]:
                    continue
                if original_state_index_2 in gfa.delta[original_state_index_1]:
                    new.addTransition(copied_state_index_1, gfa.delta[original_state_index_1][original_state_index_2], copied_state_index_2)
        subautomata.append(new)
    return subautomata

def horizontal_decomposition(gfa: GFA, state_weight, repeated) -> RegExp:
    result = None
    group = identify_group(gfa)
    subautomatas = make_horizontal_subautomata(gfa, group)
    for subautomata in subautomatas:
        if len(bridge_states(subautomata)):
            result = CDisj(result, decomposition(subautomata, state_weight, repeated))
        if state_weight and repeated:
            result = CDisj(result, repeated_state_weight_elimination(subautomata)) if result != None else repeated_state_weight_elimination(subautomata)
        elif state_weight:
            result = CDisj(result, state_weight_elimination(subautomata)) if result != None else state_weight_elimination(subautomata)
        else:
            result = CDisj(result, random_elimination(subautomata)) if result != None else random_elimination(subautomata)
    return result
    

def test_horizontal_decomposition():
    gfa = FA2GFA(nfa_with_no_bridge_and_two_group())
    horizontal_decomposition_result = horizontal_decomposition(gfa, True, False)
    print(horizontal_decomposition_result)

test_horizontal_decomposition()

'''
print(nfa)
print(nfa.States)
print(nfa.delta)
#print(FA2GFA(nfa).delta)
#nfa.display
'''

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