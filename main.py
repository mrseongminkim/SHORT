import time
import logging
import os
import random
import csv
import itertools
from pickle import load, dump

import numpy as np
import coloredlogs
from FAdo.conversions import *

from utils.data_loader import *
from utils.heuristics import *

from alpha_zero.Coach import Coach
from alpha_zero.MCTS import MCTS
from alpha_zero.utils import *
from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame as Game
from alpha_zero.state_elimination.pytorch.NNet import NNetWrapper as nn

from test import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
args = dotdict({
    'numIters': 1000,
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 100,
    'tempThreshold': 15,        # temperature hyperparameters
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.6,
    # Number of game examples to train the neural networks.
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './alpha_zero/models/',
    'load_model': False,
    'load_folder_file': ('./alpha_zero/models/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})
min_n = 3
max_n = 10
n_range = max_n - min_n + 1
alphabet = [2, 5, 10]
density = [0.2, 0.5]
sample_size = 100


def train_alpha_zero():
    log.info('Loading %s...', Game.__name__)
    g = Game()
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...',
                 args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(
            args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()
    log.info('Starting the learning process')
    c.learn()


def test_alpha_zero():
    '''
    if not os.path.isfile('./result/alpha_zero_experiment_result.pkl'):
        with open('./result/alpha_zero_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/c7.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5 - 3, 11 - 3):
                size_value = exp[n][1][0][1] / 100
                writer.writerow([size_value])
    '''
    if(1):
        data = load_data()
        exp = [[[[0, 0] for d in range(len(density))] for k in range(
            len(alphabet))] for n in range(n_range)]
        g = Game()
        nnet = nn(g)
        mcts = MCTS(g, nnet, args)
        def player(x): return np.argmax(mcts.getActionProb(x, temp=0))
        curPlayer = 1
        if args.load_model:
            nnet.load_checkpoint(args.checkpoint, args.load_folder_file[1])
        else:
            print("Can't test without pre-trained model")
            exit()
        for n in range(n_range):
            for k in range(len(alphabet)):
                for d in range(len(density)):
                    for i in range(sample_size):
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + (
                            's' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        gfa = data[n][k][d][i].dup()
                        gfa = g.getInitBoard(
                            gfa, n + min_n, alphabet[k], density[d])
                        order = []
                        start_time = time.time()
                        while g.getGameEnded(gfa, curPlayer) == -1:
                            action = player(
                                g.getCanonicalForm(gfa, curPlayer))
                            valids = g.getValidMoves(
                                g.getCanonicalForm(gfa, curPlayer), 1)
                            if valids[action] == 0:
                                assert valids[action] > 0
                            gfa, curPlayer = g.getNextState(
                                gfa, curPlayer, action)
                            order.append(action + 1)
                        result = g.gfaToBoard(gfa)[0][n + min_n + 1]
                        end_time = time.time()
                        gfa.eliminateAll(order)
                        if (result != gfa.delta[0][n + min_n + 1].treeLength()):
                            print('order', order)
                            print('result length', result)
                            print('valid length',
                                  gfa.delta[0][n + min_n + 1].treeLength())
                            print('Something is wrong')
                            exit()
                        result_time = end_time - start_time
                        exp[n][k][d][0] += result_time
                        exp[n][k][d][1] += result
        with open('./result/alpha_zero_experiment_result.pkl', 'wb') as fp:
            dump(exp, fp)


def test_heuristics():
    if os.path.isfile('./result/heuristics_experiment_result.pkl'):
        with open('./result/heuristics_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
    else:
        data = load_data()
        exp = [[[[[0, 0] for d in range(len(density))] for k in range(
            len(alphabet))] for n in range(n_range)] for c in range(6)]
        for n in range(n_range):
            for k in range(len(alphabet)):
                for d in range(len(density)):
                    for i in range(sample_size):
                        random.seed(i)
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + (
                            's' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        # eliminate_randomly
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_randomly(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[0][n][k][d][0] += result_time
                        exp[0][n][k][d][1] += result_size

                        # decompose with eliminate_randomly
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = decompose(gfa, False, False)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[1][n][k][d][0] += result_time
                        exp[1][n][k][d][1] += result_size

                        # eliminate_by_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_by_state_weight_heuristic(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[2][n][k][d][0] += result_time
                        exp[2][n][k][d][1] += result_size

                        # decompose + eliminate_by_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = decompose(gfa, True, False)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[3][n][k][d][0] += result_time
                        exp[3][n][k][d][1] += result_size

                        # eliminate_by_repeated_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_by_repeated_state_weight_heuristic(
                            gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[4][n][k][d][0] += result_time
                        exp[4][n][k][d][1] += result_size

                        # decompose + eliminate_by_repeated_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = decompose(gfa, True, True)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[5][n][k][d][0] += result_time
                        exp[5][n][k][d][1] += result_size
        with open('./result/heuristics_experiment_result.pkl', 'wb') as fp:
            dump(exp, fp)


def test_brute_force():
    data = load_data()
    exp = [[[[0, 0] for d in range(len(density))] for k in range(len(alphabet))] for n in range(n_range)]
    for n in range(3 - 3, 8 - 3):
        permutations = [x for x in range(1, n + 4)]
        for i in range(sample_size):
            print('n' + str(n + min_n) + '\'s' + str(i + 1) + 'sample')
            min_length = float('inf')
            start_time = time.time()
            for perm in itertools.permutations(permutations):
                gfa = data[n][1][0][i].dup()
                for state in perm:
                    gfa.eliminate(state)
                if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
                    length = CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]]).treeLength()
                    print('This will never run, since i specifically made GFA to not have returnining transition')
                else:
                    length = gfa.delta[gfa.Initial][list(gfa.Final)[0]].treeLength()
                min_length = min(min_length, length)
            end_time = time.time()
            exp[n][1][0][0] += end_time - start_time
            exp[n][1][0][1] += min_length
    with open('./result/brute_force_experiment_result.pkl', 'wb') as fp:
        dump(exp, fp)

def quicky():
    with open('./result/brute_force_experiment_result.pkl', 'rb') as fp:
        exp = load(fp)
    for n in range(3 - 3, 8 - 3):
        print(n + 3, exp[n][1][0][1] / 100)

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

def test_CToken():
    time_value = [0 for i in range(100)]
    length_value = [0 for i in range(100)]
    data = load_data()
    for i in range(100):
        print(i)
        gfa = data[7][2][1][i]
        start = time.time()
        result = eliminate_by_repeated_state_weight_heuristic_with_tokenization(gfa, True)
        end = time.time()
        time_value[i] = end - start
        length_value[i] = result.treeLength()
        #print(time_value[i])
    with open('./result/true_time.pkl', 'wb') as fp:
        dump(time_value, fp)
    with open('./result/true_length.pkl', 'wb') as fp:
        dump(length_value, fp)

def check_CToken():
    with open('./result/' + "true_time" + '.pkl', 'rb') as fp:
        time_value = load(fp)
    with open('./result/' + "true_length" + '.pkl', 'rb') as fp:
        length_value = load(fp)
    with open('./result/true_time.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        avg_time = 0
        for n in range(100):
            avg_time += time_value[n]
            writer.writerow([time_value[n]])
        print(avg_time / 100)
    with open('./result/true_length.csv', 'w', newline='') as fp:
        writer = csv.writer(fp)
        avg_time = 0
        for n in range(100):
            avg_time += length_value[n]
            writer.writerow([length_value[n]])
        print(avg_time / 100)
    

def main():
    train_alpha_zero()

main()
'''
data = load_data()
gfa = data[0][0][0][0]
print(gfa.States)
print(gfa.delta)
print(gfa.Initial)
print(list(gfa.Final)[0])

while (len(gfa.States) > 2):
    st = int(input('지울 스테이트 입력: '))
    eliminate_with_minimization(gfa, st)
    print(gfa.States)
    print(gfa.delta)
    print(gfa.Initial)
    print(gfa.Final)
'''