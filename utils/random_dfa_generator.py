import time
from pickle import dump

from FAdo.rndfap import *
from FAdo.fa import *

from utils.fadomata import *

seed = hash(time.perf_counter())
states = [3, 4, 5, 6, 7, 8, 9, 10]
k = 2

for n in states:
    print("n:", n)
    generator = ICDFArgen(n, k, seed=seed)
    count = 0
    content = []
    while count < 100:
        dfa: DFA = generator.next()
        if dfa.Final:
            count += 1
        else:
            continue
        print("n: %d, i: %d" % (n, count))
        dfa = dfa.minimal(complete=False)
        dfa: NFA = dfa.toNFA()
        make_nfa_complete(dfa)
        dfa = dfa.lrEquivNFA()
        dfa.renameStates()
        dfa.reorder({list(dfa.Initial)[0] : 0, 0 : list(dfa.Initial)[0], len(dfa.States) - 1 : list(dfa.Final)[0], list(dfa.Final)[0] : len(dfa.States) - 1})
        dfa.renameStates()
        dfa = convert_nfa_to_gfa(dfa)
        shuffle_fa(dfa, len(dfa.States) - 2)
        content.append(dfa)
    with open('data/random_dfa/n' + str(n) + 'k2.pkl', 'wb') as fp:
        dump(content, fp)
