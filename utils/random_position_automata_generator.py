from pickle import load, dump
import csv

from FAdo.cfg import *
from FAdo.reex import *

from utils.fadomata import *
from utils.heuristics import *

data = []
original_length = 0

length = 30
states = 15

Sigma = ['0', '1', '2', '3', '4']
regex_generator = REStringRGenerator(Sigma=Sigma, size=length)
count = 0
while count < 100:
    random_string = regex_generator.generate()
    #states = len(random_string) - random_string.count('(') - random_string.count(')')
    regular_expression = str2regexp(random_string, sigma=Sigma)
    position_automata: NFA = regular_expression_to_position_automata(regular_expression)
    if len(position_automata.States) != states:
        continue
    print('count:', count)
    original_length += regular_expression.treeLength()
    make_nfa_complete(position_automata)
    position_automata.reorder({list(position_automata.Initial)[0] : 0, 0 : list(position_automata.Initial)[0], len(position_automata.States) - 1 : list(position_automata.Final)[0], list(position_automata.Final)[0] : len(position_automata) - 1})
    position_automata.renameStates()
    #position_automata.display()
    #exit()
    gfa = convert_nfa_to_gfa(position_automata)
    #shuffle_gfa(gfa, states)
    data.append(gfa)
    count += 1

data = [data]

with open('data/position_' + str(length) + '.pkl', 'wb') as fp:
    dump(data, fp)

with open('./result/original_length_' + str(length) + '.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow([original_length / 100])
