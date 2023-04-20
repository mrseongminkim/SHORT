from pickle import load, dump

from FAdo.cfg import *
from FAdo.reex import *

from utils.fadomata import *
from utils.heuristics import *

data = [[None, None] for i in range(100)]

reverse = True
states = 20
length = 30
Sigma = ['0', '1', '2', '3', '4']
regex_generator = REStringRGenerator(Sigma=Sigma, size=length)
count = 0
while count < 100:
    random_string = regex_generator.generate()
    regular_expression = str2regexp(random_string, sigma=Sigma)
    position_automata: NFA = regular_expression_to_position_automata(regular_expression)
    if len(position_automata.States) != states:
        continue
    make_nfa_complete(position_automata)
    if reverse:
        reorder_reverse(position_automata, states)
    else:
        reorder(position_automata, states)
    gfa = convert_nfa_to_gfa(position_automata)
    data[count][0] = gfa
    data[count][1] = regular_expression.treeLength()
    count += 1

with open('data/pos_s20_l30r.pkl', 'wb') as fp:
    dump(data, fp)
