from pickle import load, dump
import csv

from FAdo.cfg import *
from FAdo.reex import *

from utils.fadomata import *
from utils.heuristics import *

data = []
original_length = 0

reverse = False
states = 7
length = 10
Sigma = ['0', '1', '2', '3', '4']
regex_generator = REStringRGenerator(Sigma=Sigma, size=length)
count = 0
while count < 100:
    random_string = regex_generator.generate()
    regular_expression = str2regexp(random_string, sigma=Sigma)
    position_automata: NFA = regular_expression_to_position_automata(regular_expression)
    if len(position_automata.States) != states:
        continue
    original_length += regular_expression.treeLength()
    make_nfa_complete(position_automata)
    if reverse:
        reorder_reverse(position_automata, states)
    else:
        reorder(position_automata, states)
    position_automata.display()
    gfa = convert_nfa_to_gfa(position_automata)

    #brute force sanity check
    permutations = [x for x in range(1, 8)]
    min_length = float('inf')
    for perm in itertools.permutations(permutations):
        temp = gfa.dup()
        for state in perm:
            temp.eliminate(state)
            #eliminate_with_minimization(temp, state, delete_state=False, tokenize=False)
        length = temp.delta[0][8].treeLength()
        min_length = min(min_length, length)
    if min_length > regular_expression.treeLength():
        print("error")
        print(temp.delta)
        print(min_length)
        print(regular_expression.treeLength())
        print(regular_expression)
        exit()
    #sanity check
    data.append(gfa)
    count += 1

data = [data]

with open('data/pos_s20_l30_r.pkl', 'wb') as fp:
    dump(data, fp)

with open('./result/original_length_r.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow([original_length / 100])