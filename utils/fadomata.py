from queue import Queue
import copy
import random

from FAdo.conversions import *
from FAdo.reex import *

from utils.CToken import *
from utils.inclusion_checker import *

def shuffle_gfa(gfa: GFA, states: int):
    order = {0 : 0, states + 1 : states + 1}
    lst = [x for x in range(1, states + 1)]
    random.shuffle(lst)
    for idx, val in enumerate(lst):
        order[idx + 1] = val
    gfa.reorder(order)


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
    if st in gfa.delta and st in gfa.delta[st]:
        r2 = copy.copy(reex.CStar(gfa.delta[st][st], copy.copy(gfa.Sigma)))
        del gfa.delta[st][st]
    else:
        r2 = None
    for s in gfa.delta:
        if st not in gfa.delta[s]:
            continue
        r1 = copy.copy(gfa.delta[s][st])
        del gfa.delta[s][st]
        for s1 in gfa.delta[st]:
            r3 = copy.copy(gfa.delta[st][s1])
            if r2 is not None:
                r = reex.CConcat(r1, reex.CConcat(r2, r3, copy.copy(gfa.Sigma)), copy.copy(gfa.Sigma))
            else:
                r = reex.CConcat(r1, r3, copy.copy(gfa.Sigma))
            if s1 in gfa.delta[s]:
                new_regex = reex.CDisj(gfa.delta[s][s1], r, copy.copy(gfa.Sigma))
                if tokenize and new_regex.treeLength() > CToken.threshold:
                    gfa.delta[s][s1] = CToken(new_regex)
                else:
                    gfa.delta[s][s1] = new_regex
            else:
                if tokenize and r.treeLength() > CToken.threshold:
                    gfa.delta[s][s1] = CToken(r)
                else:
                    gfa.delta[s][s1] = r
    if delete_state:
        gfa.deleteState(st)
    else:
        del gfa.delta[st]
    return gfa


def eliminate_with_minimization(gfa: GFA, st: int, delete_state: bool=True, tokenize: bool=True, minimize: bool=True):
    if not minimize:
        return eliminate(gfa, st, delete_state, tokenize)
    global save_count_star, save_count_concat, save_count_disj, all_count_star, all_count_concat, all_count_disj
    if st in gfa.delta and st in gfa.delta[st]:
        if isinstance(gfa.delta[st][st], CStar) or is_epsilon(gfa.delta[st][st]):
            save_count_star += 1
            r2 = copy.copy(gfa.delta[st][st])
        else:
            r2 = copy.copy(CStar(gfa.delta[st][st], copy.copy(gfa.Sigma)))
        del gfa.delta[st][st]
        all_count_star += 1
    else:
        r2 = None
    for s in gfa.delta:
        if st not in gfa.delta[s]:
            continue
        r1 = copy.copy(gfa.delta[s][st])
        del gfa.delta[s][st]
        for s1 in gfa.delta[st]:
            r3 = copy.copy(gfa.delta[st][s1])
            if r2 is not None:
                if is_included(r2, r1) == 1 or is_included(r2, r3) == 1:
                    save_count_concat += 2
                    r = CConcat(r1, r3, copy.copy(gfa.Sigma))
                elif is_epsilon(r1) and is_epsilon(r3):
                    save_count_concat += 2
                    r = r2
                elif is_epsilon(r1):
                    save_count_concat += 1
                    r = CConcat(r2, r3, copy.copy(gfa.Sigma))
                elif is_epsilon(r3):
                    save_count_concat += 1
                    r = CConcat(r1, r2, copy.copy(gfa.Sigma))
                else:
                    r = CConcat(r1, CConcat(r2, r3, copy.copy(gfa.Sigma)), copy.copy(gfa.Sigma))
            else:
                if (is_epsilon(r1) and is_epsilon(r3)):
                    save_count_concat += 2
                    r = CEpsilon()
                elif is_epsilon(r1):
                    save_count_concat += 2
                    r = r3
                elif is_epsilon(r3):
                    save_count_concat += 2
                    r = r1
                else:
                    save_count_concat += 1
                    r = CConcat(r1, r3, copy.copy(gfa.Sigma))
            all_count_concat += 2
            if s1 in gfa.delta[s]:
                check_included = is_included(r, gfa.delta[s][s1])
                if check_included == 1 or check_included == 0:
                    save_count_disj += 1
                    all_count_disj += 1
                    continue
                elif check_included == -1:
                    save_count_disj += 1
                    new_regex = r
                elif str(gfa.delta[s][s1]) > str(r):
                    new_regex = CDisj(r, gfa.delta[s][s1], copy.copy(gfa.Sigma))
                else:
                    new_regex = CDisj(gfa.delta[s][s1], r, copy.copy(gfa.Sigma))
                if tokenize and new_regex.treeLength() > CToken.threshold:
                    gfa.delta[s][s1] = CToken(new_regex)
                else:
                    gfa.delta[s][s1] = new_regex
                all_count_disj += 1
            else:
                if tokenize and r.treeLength() > CToken.threshold:
                    gfa.delta[s][s1] = CToken(r)
                else:
                    gfa.delta[s][s1] = r
    if delete_state:
        gfa.deleteState(st)
    else:
        del gfa.delta[st]
    #print_counter()
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
