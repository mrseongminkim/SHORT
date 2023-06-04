from pickle import load, dump
from os.path import isfile

from FAdo.fio import *

from utils.fadomata import *

from config import *

def load_nfa():
    data = [[] for n in range(N_RANGE)]
    for n in range(N_RANGE):
        file_name = 'n' + str(n + 3) + 'k5'
        with open('data/random_nfa/' + file_name + '.pkl', 'rb') as fp:
            data[n] = load(fp)
    return data


def load_dfa(n=None):
    if n == None:
        data = [[] for n in range(8)]
        for n in range(8):
            file_name = 'n' + str(n + 3) + 'k5'
            with open('data/random_dfa/' + file_name + '.pkl', 'rb') as fp:
                data[n] = load(fp)
        return data
    else:
        file_name = 'n' + str(n) + 'k5'
        with open('data/random_dfa/' + file_name + '.pkl', 'rb') as fp:
            data = load(fp)
        return data


'''
def load_fig10():
    nfa = NFA()
    for i in range(7):
        nfa.addState(str(i))
    nfa.setInitial([0])
    nfa.setFinal([6])
    nfa.addTransition(0, '@epsilon', 1)
    nfa.addTransition(1, '0', 2)
    nfa.addTransition(2, '1', 4)
    nfa.addTransition(3, '1', 2)
    nfa.addTransition(3, '0', 3)
    nfa.addTransition(3, '1', 3)
    nfa.addTransition(3, '2', 3)
    nfa.addTransition(3, '1', 4)
    nfa.addTransition(4, '2', 3)
    nfa.addTransition(4, '1', 5)
    nfa.addTransition(5, '@epsilon', 6)
    gfa = convert_nfa_to_gfa(nfa)
    return gfa
'''


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
    gfa = convert_nfa_to_gfa(nfa)
    return gfa


def load_position():
    with open('data/position_30.pkl', 'rb') as fp:
        data = load(fp)
    return data

def load_data(type, n=None):
    if type == 'nfa':
        return load_nfa()
    elif type == 'position':
        return load_position()
    elif type == 'fig10':
        return load_fig10()
    elif type == 'dfa':
        return load_dfa(n)
    return
