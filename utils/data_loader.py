from pickle import load, dump
from os.path import isfile

from FAdo.fio import *

from utils.fadomata import *

def load_nfa():
    data = [[] for n in range(8)]
    for n in range(8):
        file_name = 'n' + str(n + 3) + 'k5s'
        if isfile('data/random_nfa/pkl/' + file_name + '.pkl'):
            with open('data/random_nfa/pkl/' + file_name + '.pkl', 'rb') as fp:
                data[n] = load(fp)
        else:
            content = readFromFile('data/random_nfa/raw/' + file_name + '.txt')
            for i in range(len(content)):
                content[i] = convert_nfa_to_gfa(content[i])
                content[i].reorder({(content[i].States).index(x) : int(x) for x in content[i].States})
            with open('data/random_nfa/pkl/' + file_name + '.pkl', 'wb') as fp:
                dump(content, fp)
            data[n] = content
    return data
    '''
    alphabet_list = [2, 5, 10]
    density_list = ['s', 'd']
    data = [[[[] for d in range(2)] for k in range(3)] for n in range(8)]
    for n in range(3, 11):
        for k in alphabet_list:
            for d in density_list:
                a = n - 3
                b = alphabet_list.index(k)
                c = density_list.index(d)
                file_name = 'n' + str(n) + 'k' + str(k) + d
                if isfile('data/random_nfa/pkl/' + file_name + '.pkl'):
                    with open('data/random_nfa/pkl/' + file_name + '.pkl', 'rb') as fp:
                        data[a][b][c] = load(fp)
                else:
                    content = readFromFile('data/raw/' + file_name + '.txt')
                    for i in range(len(content)):
                        content[i] = convert_nfa_to_gfa(content[i])
                        content[i].reorder({(content[i].States).index(x) : int(x) for x in content[i].States})
                    with open('data/random_nfa/pkl/' + file_name + '.pkl', 'wb') as fp:
                        dump(content, fp)
                    data[a][b][c] = content
    return data
    '''


def load_position():
    min_length = 5
    max_length = 10 #max size = maxN - 1
    Sigma = ['0', '1', '2', '4', '5']
    file_name = 'n' + str(min_length) + 'to' + str(max_length - 1) + 'k' + str(len(Sigma)) + '.pkl'
    with open('data/position_automata/' + file_name, 'rb') as fp:
        data = load(fp)
    return data


def load_fig10():
    nfa = NFA()
    for i in range(5):
        nfa.addState(str(i))
    nfa.setInitial([0])
    nfa.setFinal([4])
    nfa.addTransition(0, '0', 1)
    nfa.addTransition(1, '1', 3)
    nfa.addTransition(2, '1', 1)
    nfa.addTransition(2, '0', 2)
    nfa.addTransition(2, '1', 2)
    nfa.addTransition(2, '2', 2)
    nfa.addTransition(2, '1', 3)
    nfa.addTransition(3, '2', 2)
    nfa.addTransition(3, '1', 4)
    #nfa.display()
    gfa = convert_nfa_to_gfa(nfa)
    return gfa

def load_data(type):
    if type == 'nfa':
        return load_nfa()
    elif type == 'position':
        return load_position()
    elif type == 'fig10':
        return load_fig10()
    return
