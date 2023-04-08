import time

from FAdo.rndfap import *
from FAdo.fa import *

from utils.fadomata import *

seed = hash(time.perf_counter())

n = 7
k = 2
generator = ICDFArgen(n, k, seed=seed)

count = 0
while (count < 100):
    dfa: DFA = generator.next()
    if dfa.Final:
        count += 1
    else:
        continue
    nfa = dfa.toNFA()
    make_nfa_complete(nfa)
    order = {}
    for i in range(len(nfa.States)):
        if i == len(nfa.States) - 2:
            order[i] = 0
        elif i == len(nfa.States) - 1:
            order[i] = i
        else:
            order[i] = i + 1
    nfa.reorder(order)
    nfa.renameStates()
    gfa = convert_nfa_to_gfa(nfa)
    exit()
