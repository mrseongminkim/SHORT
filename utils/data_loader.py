from pickle import load, dump
from os.path import isfile

from FAdo.fio import *

from utils.fadomata import *

def load_data() -> list:
    alphabet_list = [2, 5, 10]
    density_list = ['s', 'd']
    data = [[[[0] for d in range(2)] for k in range(3)] for n in range(8)]
    for n in range(3, 11):
        for k in alphabet_list:
            for d in density_list:
                a = n - 3
                b = alphabet_list.index(k)
                c = density_list.index(d)
                file_name = 'data/n' + str(n) + 'k' + str(k) + d
                if isfile(file_name + '.pickle'):
                    with open(file_name + '.pickle', 'rb') as fp:
                        data[a][b][c] = load(fp)
                else:
                    content = readFromFile(file_name + '.txt')
                    for i in range(len(content)):
                        content[i] = convert_nfa_to_gfa(content[i])
                        content[i].reorder({(content[i].States).index(x) : int(x) for x in content[i].States})
                    with open(file_name + '.pickle', 'wb') as fp:
                        dump(content, fp)
                    data[a][b][c] = content
    return data