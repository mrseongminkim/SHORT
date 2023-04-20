from pickle import load, dump

from FAdo.fl import *

from utils.fadomata import *

data = []

Sigma = ['0', '1', '2', '3', '4']
count = 0
while count < 100:
    print("count: ", count)
    dfa = genRndTrieUnbalanced(5, Sigma, 0.5)
    dfa = dfa.toNFA()
    if len(dfa.States) != 50:
        continue
    make_nfa_complete(dfa)
    order = {}
    for j in range(len(dfa.States)):
        if j == len(dfa.States) - 2:
            order[j] = 0
        elif j == len(dfa.States) - 1:
            order[j] = j
        else:
            order[j] = j + 1
    #dfa.display()
    dfa.reorder(order)
    dfa.renameStates()

    n = 50
    order = {0 : 0, 51 : 51}
    queue = Queue()
    queue.put(1)
    while not queue.empty():
        curr = queue.get()
        order[curr] = n
        n -= 1
        for x in list(dfa.delta[curr].values()):
            for element in x:
                if element not in order:
                    queue.put(element)
    #n = 1
    #n += 1
    #to make left to right order
    dfa = convert_nfa_to_gfa(dfa)
    data.append(dfa)
    count += 1

with open('data/trie.pkl', 'wb') as fp:
    dump(data, fp)
