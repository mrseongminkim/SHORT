from pickle import load, dump

from FAdo.cfg import *
from FAdo.reex import *

from utils.fadomata import *

data = [[None, None] for i in range(100)]

length = 92
Sigma = ['0', '1', '2', '3', '4']
regex_generator = REStringRGenerator(Sigma=Sigma, size=length)
count = 0
while count < 100:
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
    if len(gfa.States) == 52:
        print("count:", count)
        data[count][0] = gfa
        data[count][1] = regular_expression.treeLength()
        count += 1

with open('data/position_automata.pkl', 'wb') as fp:
    dump(data, fp)
