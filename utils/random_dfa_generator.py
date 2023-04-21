import time
from pickle import dump

from FAdo.rndfap import *
from FAdo.fa import *

from utils.fadomata import *

seed = hash(time.perf_counter())

states = [3, 4, 5, 6, 7, 8, 9, 10]
k = 5

for n in states:
    print("n:", n)
    generator = ICDFArgen(n, k, seed=seed)
    count = 0
    content = []
    while count < 100:
        print('what')
        dfa: DFA = generator.next()
        if dfa.Final:
            count += 1
        else:
            continue
        dfa = dfa.toNFA()
        make_nfa_complete(dfa)
        reorder(dfa, n)
        content.append(convert_nfa_to_gfa(dfa))
    file_name = "n" + str(n) + "k5.pkl"
    with open('data/random_dfa/' + file_name, 'wb') as fp:
        dump(content, fp)

print('done')
