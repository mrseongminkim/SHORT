import copy

from FAdo.conversions import *
from FAdo.reex import *

#str(re) is costly operation
def is_epsilon(regex: RegExp):
    return isinstance(regex, CEpsilon)


'''
    typo: in_included -> is_included
    elif is_epsilon(re1) and is_epsilon(re2):
        return 0
        -> same as re1 == re2
    elif isinstance(re1, CConcat) and isinstance(re2, CConcat):
        if (re1.arg1 == re2.arg1) and (re1.arg2 == re2.arg2):
            return 0
        -> same as re1 == re2
'''
def is_included(re1: RegExp, re2: RegExp):
    """
    if re1 is included in re2, return 1
    if re1 is equivalent to re2, return 0
    if re2 is included in re1, return -1
    if not, return 2
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

#Trace Values
save_count_star = 0
save_count_concat = 0
save_count_disj = 0
all_count_star = 0
all_count_concat = 0
all_count_disj = 0

#Counterpart of GFA.eliminate method
def eliminate(gfa: GFA, st: int):
    global save_count_star, save_count_concat, save_count_disj, all_count_star, all_count_concat, all_count_disj
    #Finiding r1 r2* r3
    #r2
    if st in gfa.delta and st in gfa.delta[st]:
        if isinstance(gfa.delta[st][st], CStar) or is_epsilon(gfa.delta[st][st]):
            #Trace
            save_count_star += 1
            r2 = copy.copy(gfa.delta[st][st])
        else:
            r2 = copy.copy(CStar(gfa.delta[st][st], copy.copy(gfa.Sigma)))
        del gfa.delta[st][st]
        #Trace
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
                #???????????????????????????????????????////////??????????????????????????????
                if is_included(r2, r1) == 1 or is_included(r2, r3) == 1:
                    save_count_concat += 1
                    #Change it to token
                    r = CConcat(r1, r3, copy.copy(gfa.Sigma))
                elif is_epsilon(r1):
                    save_count_concat += 1
                    if is_epsilon(r3):
                        r = r2
                    else:
                        #Change it to token
                        r = CConcat(r2, r3, copy.copy(gfa.Sigma))
                elif is_epsilon(r3):
                    save_count_concat += 1
                    #Change it to token
                    r = CConcat(r1, r2, copy.copy(gfa.Sigma))
                else:
                    #Change it to token
                    r = CConcat(r1, CConcat(r2, r3, copy.copy(gfa.Sigma)), copy.copy(gfa.Sigma))
            else:
                if (is_epsilon(r1) and is_epsilon(r3)):
                    save_count_concat += 2
                    r = CEpsilon()
                elif is_epsilon(r1):
                    save_count_concat += 1
                    r = r3
                elif is_epsilon(r3):
                    save_count_concat += 1
                    r = r1
                else:
                    #Change it to token
                    r = CConcat(r1, r3, copy.copy(gfa.Sigma))
            all_count_concat += 1

            #s as source state, s1 as target state
            #basically, transition label already exsits
            if s1 in gfa.delta[s]:
                # print(f"R1: {r}, R2: {gfa.delta[s][s1]}")
                check_included = is_included(r, gfa.delta[s][s1])
                if check_included == 1 or check_included == 0:
                    save_count_disj += 1
                    gfa.delta[s][s1] = r #?????gfa.delta[s][s1]이 되어야 하는거 아닌가?
                elif check_included == -1:
                    save_count_disj += 1
                    gfa.delta[s][s1] = gfa.delta[s][s1] #불필요
                else:
                    #Change it to token
                    if str(gfa.delta[s][s1]) > str(r):
                        gfa.delta[s][s1] = CDisj(
                            r, gfa.delta[s][s1], copy.copy(gfa.Sigma))
                    else:
                        gfa.delta[s][s1] = CDisj(
                            gfa.delta[s][s1], r, copy.copy(gfa.Sigma))
                all_count_disj += 1
            else:
                #Change it to toekn
                gfa.delta[s][s1] = r
    del gfa.delta[st]
    return gfa

class CToken(RegExp):
    #Static class variable
    token_to_regex = dict()

    def __init__(self, regex: RegExp):
        self.hashed_value = hash(regex)
        #Sanity Check
        if self.hashed_value in CToken.token_to_regex:
            assert CToken.token_to_regex[self.hashed_value] == regex
        #Sanity Check
        CToken.token_to_regex[self.hashed_value] = regex
    
    def __str__(self):
        return str(CToken.token_to_regex[self.hashed_value])
    
    def __repr__(self):
        return repr(CToken.token_to_regex[self.hashed_value])

    def treeLength(self):
        return CToken.token_to_regex[self.hashed_value].treeLength()

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

regex = str2regexp("(3 + @epsilon)*")
print(repr(regex))
x = is_epsilon(regex)
print(x)