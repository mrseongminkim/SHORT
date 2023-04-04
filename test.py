from FAdo.reex import *

from utils.heuristics import *

def eliminate_by_repeated_state_weight_heuristic_with_tokenization(gfa: GFA, tokenize: bool=False) -> RegExp:
    n = len(gfa.States) - 2
    victim = [i + 1 for i in range(len(gfa.States) - 2)]
    for i in range(n):
        if (len(victim) == 1):
            eliminate_with_tokenization(gfa, victim[0], tokenize)
            continue
        min_val = get_weight(gfa, victim[0])
        min_idx = 0
        for j in range(1, len(victim)):
            curr_val = get_weight(gfa, victim[j])
            if min_val > curr_val:
                min_val = curr_val
                min_idx = j
        eliminate_with_tokenization(gfa, victim[min_idx], tokenize)
        del victim[min_idx]
    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
        return CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]])
    else:
        return gfa.delta[gfa.Initial][list(gfa.Final)[0]]


def eliminate_with_tokenization(gfa: GFA, st: int, tokenize: bool=False):
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
    del gfa.delta[st]

#deprecated -> utils.fadomata.eliminate_with_minimization
'''
from FAdo.reex import *


def is_terminal(regex: RegExp) -> bool:
    return isinstance(regex, CEpsilon) or isinstance(regex, CAtom)

# return minimized regular expression and list of independent items as tuple (internally)
# independent items: CAtom, CEpsilon, and CStar


def minimize_regular_expression(regex: RegExp) -> tuple:
    # base case
    if is_terminal(regex):
        return regex, [regex]

    # recursive case
    if isinstance(regex, CStar):
        arg, arg_items = minimize_regular_expression(regex.arg)
        # Case: (R*)* = R*
        if isinstance(arg, CStar):
            return arg, [arg]
        else:
            return CStar(arg), [CStar(arg)]

    elif isinstance(regex, CDisj):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)
        # Case: R + R = R and hopefully catches (R1 + R2) + R3 = R1 + (R2 + R3)
        if set(first_items) == set(second_items):
            return first, first_items
        else:
            if isinstance(first, CConcat):
                first_items = [first]
            if isinstance(second, CConcat):
                second_items = [second]
            return CDisj(first, second), first_items + second_items

    elif isinstance(regex, CConcat):
        first, first_items = minimize_regular_expression(regex.arg1)
        second, second_items = minimize_regular_expression(regex.arg2)
        # Case: (R1 ⋅ R2) ⋅ R3 = R1 ⋅ (R2 ⋅ R3)
        return CConcat(first, second), first_items + second_items

    else:
        print('further improvement?')
        return regex, [regex]
'''