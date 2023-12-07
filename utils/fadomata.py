from queue import Queue
import copy
import random

from FAdo.conversions import *
from FAdo.reex import *

from utils.CToken import *
from utils.inclusion_checker import *

from config import *

def reverse_gfa(gfa: GFA):
    rev = GFA()
    rev.setSigma(gfa.Sigma)
    rev.States = gfa.States[:]
    rev.setFinal([gfa.Initial])
    rev.setInitial(list(gfa.Final)[0])
    rev.predecessors = {}
    for i in range(len(gfa.States)):
        rev.predecessors[i] = set([])
        rev.delta[i] = {}
    for source in gfa.delta:
        for target in gfa.delta[source]:
            rev.predecessors[source].add(target)
            rev.delta[target][source] = copy.deepcopy(gfa.delta[source][target])
    return rev

def rename_states(fa):
    #0은 존재하지 않는 것, init과 final 또한 MAX_STATES + 2 이하인 정수로 매핑됨
    lst = [str(i) for i in range(1, MAX_STATES + 3)]
    sampled_states_number = random.sample(lst, len(fa.States))
    garbage_name = ['init' for _ in range(len(fa.States))]
    fa.renameStates(garbage_name)
    fa.renameStates(sampled_states_number)
    return fa

def shuffle_fa(fa):
    order = {}
    lst = [i for i in range(len(fa.States))]
    random.shuffle(lst)
    for idx, val in enumerate(lst):
        order[idx] = val
    fa.reorder(order)
    return fa

#obsoleted
'''
def reorder(nfa: NFA, states: int, skip_first_sort=False):
    if not skip_first_sort:
        order = {}
        for j in range(len(nfa.States)):
            if j == len(nfa.States) - 2:
                order[j] = 0
            elif j == len(nfa.States) - 1:
                order[j] = j
            else:
                order[j] = j + 1
        nfa.reorder(order)
        nfa.renameStates()
    order = {0 : 0, states + 1 : states + 1}
    visited = [0, 1, states + 1]
    queue = Queue()
    queue.put(1)
    n = 1
    while not queue.empty():
        curr = queue.get()
        order[curr] = n
        n += 1
        if curr not in nfa.delta:
            continue
        for x in list(nfa.delta[curr].values()):
            for element in x:
                if element not in visited:
                    queue.put(element)
                    visited.append(element)
    nfa.reorder(order)
    nfa.renameStates()


def reorder_reverse(nfa: NFA, states: int):
    order = {}
    for j in range(len(nfa.States)):
        if j == len(nfa.States) - 2:
            order[j] = 0
        elif j == len(nfa.States) - 1:
            order[j] = j
        else:
            order[j] = j + 1
    nfa.reorder(order)
    nfa.renameStates()
    order = {0 : 0, states + 1 : states + 1}
    visited = [0, 1, states + 1]
    queue = Queue()
    queue.put(1)
    n = states
    while not queue.empty():
        curr = queue.get()
        order[curr] = n
        n -= 1
        if curr not in nfa.delta:
            continue
        for x in list(nfa.delta[curr].values()):
            for element in x:
                if element not in visited:
                    queue.put(element)
                    visited.append(element)
    nfa.reorder(order)
    nfa.renameStates()
'''

save_count_star = 0
save_count_concat = 0
save_count_disj = 0
all_count_star = 0
all_count_concat = 0
all_count_disj = 0
def print_counter():
    global save_count_star, save_count_concat, save_count_disj, all_count_star, all_count_concat, all_count_disj
    print("saved star: ", all_count_star - save_count_star)
    print("saved concat: ", all_count_concat - save_count_concat)
    print("saved disj: ", all_count_disj - save_count_disj)
    save_count_star = 0
    save_count_concat = 0
    save_count_disj = 0
    all_count_star = 0
    all_count_concat = 0
    all_count_disj = 0


def eliminate(gfa: GFA, st: int, delete_state: bool=True, tokenize: bool=True):
    #i as a intransition node
    #j as a outtransition node
    for i in gfa.predecessors[st]:
        for j in gfa.delta[st]:
            if i != st and j != st:
                #in transition
                rex = gfa.delta[i][st]
                #self loop
                if st in gfa.delta[st]:
                    rex = reex.CConcat(rex, reex.CStar(gfa.delta[st][st], copy(gfa.Sigma)), copy(gfa.Sigma))
                #out transition
                rex = reex.CConcat(rex, gfa.delta[st][j], copy(gfa.Sigma))
                #if there was already transition
                if j in gfa.delta[i]:
                    rex = reex.CDisj(gfa.delta[i][j], rex, copy(gfa.Sigma))
                if tokenize and rex.treeLength() > CToken.threshold:
                    gfa.delta[i][j] = CToken(rex)
                else:
                    gfa.delta[i][j] = rex
                #deleting st from predecessors happens in deleteState
                gfa.predecessors[j].add(i)
    if delete_state:
        gfa.deleteState(st)
    else:
        del gfa.delta[st]
    return gfa

#이제 여기에 predecessor를 추가하면 됩니다 ㅎ
def eliminate_with_minimization(gfa: GFA, st: int, delete_state: bool=True, tokenize: bool=True, minimize: bool=True):
    if not minimize:
        return eliminate(gfa, st, delete_state, tokenize)
    self_loop = None
    if st in gfa.delta[st]:
        skip_star = int(isinstance(gfa.delta[st][st], CStar) or is_epsilon(gfa.delta[st][st]))
        save_count_star += skip_star
        self_loop = copy.copy(gfa.delta[st][st]) if skip_star else copy.copy(CStar(gfa.delta[st][st], copy.copy(gfa.Sigma)))
        del gfa.delta[st][st]
        gfa.predecessors[st].remove(st)
        all_count_star += 1
    for i in gfa.predecessors[st]:
        in_transition = copy.copy(gfa.delta[i][st])
        for j in gfa.delta[st]:
            out_transition = copy.copy(gfa.delta[st][j])
            if self_loop:
                if is_included(self_loop, in_transition) == 1 or is_included(self_loop, out_transition) == 1:
                    save_count_concat += 2
                    r = CConcat(in_transition, out_transition, copy.copy(gfa.Sigma))
                elif is_epsilon(in_transition) and is_epsilon(out_transition):
                    save_count_concat += 2
                    r = self_loop
                elif is_epsilon(in_transition):
                    save_count_concat += 1
                    r = CConcat(self_loop, out_transition, copy.copy(gfa.Sigma))
                elif is_epsilon(out_transition):
                    save_count_concat += 1
                    r = CConcat(in_transition, self_loop, copy.copy(gfa.Sigma))
                else:
                    r = CConcat(CConcat(in_transition, self_loop, copy.copy(gfa.Sigma)), out_transition, copy.copy(gfa.Sigma))
            else:
                if is_epsilon(in_transition) and is_epsilon(out_transition):
                    save_count_concat += 2
                    r = CEpsilon()
                elif is_epsilon(in_transition):
                    save_count_concat += 2
                    r = out_transition
                elif is_epsilon(out_transition):
                    save_count_concat += 2
                    r = in_transition
                else:
                    save_count_concat += 1
                    r = CConcat(in_transition, out_transition, copy.copy(gfa.Sigma))
            all_count_concat += 2
            if j in gfa.delta[i]:
                check_included = is_included(r, gfa.delta[i][j])
                if check_included == 1 or check_included == 0:
                    save_count_disj += 1
                    all_count_disj += 1
                    continue
                elif check_included == -1:
                    save_count_disj += 1
                elif str(gfa.delta[i][j]) > str(r):
                    r = CDisj(r, gfa.delta[i][j], copy.copy(gfa.Sigma))
                else:
                    r = CDisj(gfa.delta[i][j], r, copy.copy(gfa.Sigma))
                if tokenize and r.treeLength() > CToken.threshold:
                    gfa.delta[i][j] = CToken(r)
                else:
                    gfa.delta[i][j] = r
                all_count_disj += 1
            else:
                if tokenize and r.treeLength() > CToken.threshold:
                    gfa.delta[i][j] = CToken(r)
                else:
                    gfa.delta[i][j] = r
            gfa.predecessors[j].add(i)
    if delete_state:
        gfa.deleteState(st)
    else:
        del gfa.delta[st]
    return gfa


def make_nfa_complete(nfa):
    initial = nfa.addState()
    for i in nfa.Initial:
        nfa.addTransition(initial, Epsilon, i)
    nfa.setInitial([initial])
    final = nfa.addState()
    for i in nfa.Final:
        nfa.addTransition(i, Epsilon, final)
    nfa.setFinal([final])


#Counterpart of RegExp.nfaGlushkov
def regular_expression_to_position_automata(regular_expression):
    aut = fa.NFA()
    initial = aut.addState('0')
    aut.addInitial(initial)
    if regular_expression.Sigma is not None:
        aut.setSigma(regular_expression.Sigma)
    _, final = regular_expression._nfaGlushkovStep(aut, aut.Initial, set())
    aut.Final = final
    aut.epsilon_transitions=  False
    return aut


#Counterpart of GFA.weight method
def get_weight(gfa: GFA, state: int) -> int:
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


#Counterpart of FA2GFA function
def convert_nfa_to_gfa(nfa: NFA) -> GFA:
    gfa = GFA()
    gfa.setSigma(nfa.Sigma)
    gfa.Initial = uSet(nfa.Initial)
    gfa.States = nfa.States[:]
    gfa.setFinal(nfa.Final)
    gfa.predecessors = {}
    for i in range(len(gfa.States)):
        gfa.predecessors[i] = set([])
    for s in nfa.delta:
        for c in sorted(nfa.delta[s].keys(), reverse=False):
            for s1 in nfa.delta[s][c]:
                gfa.addTransition(s, c, s1)
    for i in range(len(gfa.States)):
        if i not in gfa.delta:
            gfa.delta[i] = {}
    return gfa


#Counterpart of GFA.addTransition
def add_transition(gfa: GFA, sti1: int, sym: RegExp, sti2: int):
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
