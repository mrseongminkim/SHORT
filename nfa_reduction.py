from pickle import load, dump
from os.path import isfile

from FAdo.fio import *

from utils.fadomata import *

def test_reduction() -> list:
    alphabet_list = [5]
    density_list = ['s']



    data = [[[[0, 0, 0] for d in range(1)] for k in range(1)] for n in range(5)]
    for n in range(3, 8):
        for k in alphabet_list:
            for d in density_list:
                file_name = 'n' + str(n) + 'k' + str(5) + "s"
                content = readFromFile('data/raw/' + file_name + '.txt')
                for i in range(len(content)):
                    print(n, k, d, i)
                    nfa = content[i].dup()
                    nfa.reorder({(content[i].States).index(x) : int(x) for x in content[i].States})

                    temp = nfa.dup()
                    data[n - 3][0][0][0] += len(temp.rEquivNFA().States)

                    temp = nfa.dup()
                    data[n - 3][0][0][1] += len(temp.lEquivNFA().States)

                    temp = nfa.dup()
                    data[n - 3][0][0][2] += len(temp.lrEquivNFA().States)
                    for q in range(3):
                        print(data[n - 3][0][0][q], end = "\t")

    with open('data/' + "reduction" + '.pkl', 'wb') as fp:
        dump(data, fp)

def check_result():
    data = list()
    with open('data/' + "reduction" + '.pkl', 'rb') as fp:
        data = load(fp)
    for n in range(3, 8):
        print(data[n - 3][0][0])

check_result()