import copy

from FAdo.conversions import *
from FAdo.reex import *

from utils.CToken import *














#'''
def is_included(re1: RegExp, re2: RegExp, depth=0):
    MAX_RECURSION_DEPTH = 5
    depth += 1
    if depth > MAX_RECURSION_DEPTH:
        return 2
    if is_epsilon(re1):
        #this is same as re1 == re2
        if is_epsilon(re2):
            return 0
        elif isinstance(re2, CDisj):
            left = is_included(re1, re2.arg1, depth)
            right = is_included(re1, re2.arg2, depth)
            if left != 2 or right != 2:
                return 1
            else:
                return 2
        #CConcat can't include anything
        elif isinstance(re2, CConcat):
            return 2
        #star of any regex always includes epsilon
        elif isinstance(re2, CStar):
            return 1
        elif isinstance(re2, CToken):
            return isinstance(re1, CToken.token_to_regex[re2.hashed_value], depth)
    elif is_epsilon(re2):
        val = - is_included(re2, re1, depth - 1)
        return 2 if val == -2 else val

    if isinstance(re1, CDisj):
        #if is_epsilon(re2); this is done by: elif is_epsilon(re2):
        if isinstance(re2, CDisj):
            first_argument = is_included(re1.arg1, re2.arg1, depth) == 1 or is_included(re1.arg1, re2.arg2, depth) == 1
            second_argument = is_included(re1.arg2, re2.arg1, depth) == 1 or is_included(re1.arg2, re2.arg2, depth) == 1
            if first_argument and second_argument:
                return 1
            else:
                return 2
        elif isinstance(re2, CConcat) or isinstance(re2, CStar):
            first_argument = is_included(re1.arg1, re2, depth)
            second_argument = is_included(re1.arg2, re2, depth)
            if first_argument == 1 or second_argument ==  1:
                return 1
            else:
                return 2
        elif isinstance(re2, CToken):
            first_argument = isinstance(re1.arg1, CToken.token_to_regex[re2.hashed_value], depth)

#'''



















def is_epsilon(regex: RegExp):
    return isinstance(regex, CEpsilon)


#will be replaced by better(i hope, i pray) function
def is_included(re1: RegExp, re2: RegExp):
    """
    if re1 ⊆ re2:
        return 1
    if re1 == re2:
        return 0
    if re2 ⊆ re1:
        return -1
    else:
        return 2
    """
    if re1 == re2:
        return 0
    elif is_epsilon(re1) and isinstance(re2, CStar):
        return 1
    elif is_epsilon(re2) and isinstance(re1, CStar):
        return -1
    elif isinstance(re2, CDisj) and (re2.arg1 == re1 or re2.arg2 == re1):
        return 1
    elif isinstance(re1, CDisj) and (re1.arg1 == re2 or re1.arg2 == re2):
        return -1
    elif isinstance(re1, CDisj) and isinstance(re2, CDisj):
        if (is_included(re1.arg1, re2.arg2) == 0) and (is_included(re1.arg2, re2.arg1) == 0):
            return 0
    elif isinstance(re1, CStar) and isinstance(re2, CStar):
        return is_included(re1.arg, re2.arg)
    return 2


def eliminate_with_tokenization(gfa: GFA, st: int, tokenize: bool=True, delete_state=True):
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
                r = reex.CDisj(gfa.delta[s][s1], r, copy.copy(gfa.Sigma))
            if tokenize and r.treeLength() > CToken.threshold:
                gfa.delta[s][s1] = CToken(r)
            else:
                gfa.delta[s][s1] = r
    if delete_state:
        gfa.deleteState(st)
    else:
        del gfa.delta[st]
    return gfa


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


def eliminate_with_minimization(gfa: GFA, st: int, delete_state: bool=True, tokenize: bool=True):
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
                    if delete_state:
                        gfa.deleteState(st)
                    else:
                        del gfa.delta[st]
                    all_count_disj += 1
                    #print_counter()
                    return gfa
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
