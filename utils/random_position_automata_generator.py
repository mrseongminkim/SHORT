from pickle import load, dump

from FAdo.cfg import *
from FAdo.reex import *

from utils.fadomata import *

#5, 6, 7, 8, 9; total 5
data = [[[None, None] for i in range(100)] for size in range(0, 5)]

min_length = 5
max_length = 10 #max size = maxN - 1
Sigma = ['0', '1', '2', '4', '5']
for size in range(min_length, max_length):
    regex_generator = REStringRGenerator(Sigma=Sigma, size=size)
    for i in range(100):
        random_string = regex_generator.generate()
        regular_expression = str2regexp(random_string, sigma=Sigma)
        position_automata: NFA = regular_expression_to_position_automata(regular_expression)
        make_nfa_complete(position_automata)
        order = {}
        for j in range(len(position_automata.States)):
            if j == len(position_automata.States) - 2:
                order[j] = 0
            elif j == len(position_automata.States) - 1:
                order[j] = j
            else:
                order[j] = j + 1
        position_automata.reorder(order)
        position_automata.renameStates()
        gfa = convert_nfa_to_gfa(position_automata)
        data[size - min_length][i][0] = gfa
        data[size - min_length][i][1] = regular_expression.treeLength()

file_name = 'n' + str(min_length) + 'to' + str(max_length - 1) + 'k' + str(len(Sigma)) + '.pkl'
with open('data/position_automata/' + file_name, 'wb') as fp:
    dump(data, fp)
