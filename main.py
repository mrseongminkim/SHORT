import time
import logging
import os
import random
import csv
import itertools
from pickle import load, dump

import torch
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

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
args = dotdict({
    'numIters': 1000,
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 100,
    #'tempThreshold': 4,        # temperature hyperparameters
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.0,
    # Number of game examples to train the neural networks.
    'maxlenOfQueue': 200000,
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 40,
    'cpuct': 2,
    'checkpoint': './alpha_zero/models/deleting/',
    'load_model': True,
    'load_folder_file': ('./alpha_zero/models/deleting/', 'n50_iter26_sims50_cpunct2.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})
min_n = 3
max_n = 10
n_range = max_n - min_n + 1
alphabet = 5
density = 0.2
sample_size = 100

def train_alpha_zero():
    print("Let's briefly check the important hyperparameters.")
    print("\tnumMCTSSims: ", args.numMCTSSims)
    print("\tcpuct: ", args.cpuct)
    print("\tcuda: ", torch.cuda.is_available())
    log.info('Loading %s...', Game.__name__)
    g = Game()
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    if not args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)
    if not args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()
    log.info('Starting the learning process')
    c.learn()


def test_alpha_zero(model_updated):
    model_updated = model_updated
    if not model_updated and os.path.isfile('./result/alpha_zero_experiment_result.pkl'):
        with open('./result/alpha_zero_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/rl_length.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(n_range):
                size_value = exp[n][0] / 100
                writer.writerow([size_value])
        with open('./result/rl_time.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(n_range):
                time_value = exp[n][1] / 100
                writer.writerow([time_value])
    else:
        data = load_data('nfa')
        exp = [[0, 0] for n in range(n_range)]
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
            for i in range(sample_size):
                print('n: ' + str(n + min_n) + ', i:', i)
                gfa = data[n][i]
                gfa = g.getInitBoard(gfa, n + min_n, 5, 0.2)
                start_time = time.time()
                while g.getGameEnded(gfa, curPlayer) == None:
                    action = player(g.getCanonicalForm(gfa, curPlayer))
                    valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                    if valids[action] == 0:
                        assert valids[action] > 0
                    gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
                end_time = time.time()
                result_length = gfa.delta[0][1].treeLength()
                result_time = end_time - start_time
                exp[n][0] += result_length
                exp[n][1] += result_time
        with open('./result/alpha_zero_experiment_result.pkl', 'wb') as fp:
            dump(exp, fp)


def test_heuristics(model_updated):
    model_updated = False
    if not model_updated and os.path.isfile('./result/heuristics_experiment_result.pkl'):
        with open('./result/heuristics_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        for c in range(6):
            with open('./result/c' + str(c + 1) + '_length.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                for n in range(n_range):
                    size_value = exp[c][n][0] / 100
                    writer.writerow([size_value])
            with open('./result/c' + str(c + 1) + '_time.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                for n in range(n_range):
                    time_value = exp[c][n][1] / 100
                    writer.writerow([time_value])
    else:
        data = load_data('nfa')
        exp = [[[0, 0] for n in range(n_range)] for c in range(6)]
        for n in range(n_range):
            for i in range(sample_size):
                print('n: ' + str(n + min_n) + ', i:', i)
                # eliminate_randomly
                gfa = data[n][i].dup()
                start_time = time.time()
                result = eliminate_randomly(gfa)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[0][n][0] += result_size
                exp[0][n][1] += result_time

                # decompose with eliminate_randomly
                gfa = data[n][i].dup()
                start_time = time.time()
                result = decompose(gfa, False, False)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[1][n][0] += result_size
                exp[1][n][1] += result_time

                # eliminate_by_state_weight_heuristic
                gfa = data[n][i].dup()
                start_time = time.time()
                result = eliminate_by_state_weight_heuristic(gfa)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[2][n][0] += result_size
                exp[2][n][1] += result_time

                # decompose + eliminate_by_state_weight_heuristic
                gfa = data[n][i].dup()
                start_time = time.time()
                result = decompose(gfa, True, False)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[3][n][0] += result_size
                exp[3][n][1] += result_time

                # eliminate_by_repeated_state_weight_heuristic
                gfa = data[n][i].dup()
                start_time = time.time()
                result = eliminate_by_repeated_state_weight_heuristic(gfa)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[4][n][0] += result_size
                exp[4][n][1] += result_time

                # decompose + eliminate_by_repeated_state_weight_heuristic
                gfa = data[n][i].dup()
                start_time = time.time()
                result = decompose(gfa, True, True)
                end_time = time.time()
                result_time = end_time - start_time
                result_size = result.treeLength()
                exp[5][n][0] += result_size
                exp[5][n][1] += result_time
        with open('./result/heuristics_experiment_result.pkl', 'wb') as fp:
            dump(exp, fp)


def test_alpha_zero_for_position_automata(model_updated):
    model_updated = model_updated
    if not model_updated and os.path.isfile('./result/alpha_zero_position_result.pkl'):
        with open('./result/alpha_zero_position_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/rl_position.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([exp])
        with open('./result/postion_original_length.pkl', 'rb') as fp:
            original = load(fp)
        with open('./result/original_position.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerow([original])
    else:
        data = load_data('position')
        exp = 0
        original = 0
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
        for i in range(100):
            print("i:", i)
            gfa = data[i][0]
            original_length = data[i][1]
            gfa = g.getInitBoard(gfa, len(gfa.States) - 2)
            while g.getGameEnded(gfa, curPlayer) == None:
                action = player(g.getCanonicalForm(gfa, curPlayer))
                valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                if valids[action] == 0:
                    assert valids[action] > 0
                gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
            exp += gfa.delta[0][1].treeLength()
            original += original_length
        exp /= 100
        original /= 100
        with open('./result/alpha_zero_position_result.pkl', 'wb') as fp:
            dump(exp, fp)
        with open('./result/postion_original_length.pkl', 'wb') as fp:
            dump(original, fp)


#not working
def test_brute_force():
    model_updated = True
    if not model_updated and os.path.isfile('./result/brute_force_experiment_result.pkl'):
        with open('./result/brute_force_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/c8.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5):
                #k = 5, d = 0.2
                size_value = exp[n][1][0][1] / 100
                writer.writerow([size_value])
    else:
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


#not working
def test_reduction():
    model_updated = True
    if not model_updated and os.path.isfile('data/reduction.pkl'):
        with open('data/reduction.pkl', 'rb') as fp:
            data = load(fp)
        for n in range(3, 8):
            print(data[n - 3][0][0])
    else:
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
        with open('data/reduction.pkl', 'wb') as fp:
            dump(data, fp)


#not working
def test_fig10():
    gfa = load_data('fig10')
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
    order = []
    gfa = g.getInitBoard(gfa, len(gfa.States) - 2)
    while g.getGameEnded(gfa, curPlayer) == None:
        action = player(g.getCanonicalForm(gfa, curPlayer))
        valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
        if valids[action] == 0:
            assert valids[action] > 0
        order.append(gfa.States[action])
        gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
    print(order)


def main():
    print("deleting-states")
    test_alpha_zero_for_position_automata(True)
    test_alpha_zero_for_position_automata(False)
    #train_alpha_zero()
    #test_alpha_zero(True)
    #test_alpha_zero(False)
    #print("test-heuristics")
    #test_heuristics(True)
    #test_heuristics(False)


main()

#'''
from utils.random_nfa_generator import *
import time

def divide_and_conquer(n):
    length_list = []
    for i in range(100):
        permutations = [x for x in range(1, 8)]
        gfa = generate(7, 5, 0.1, 'in-memory')
        min_length = float('inf')
        for perm in itertools.permutations(permutations):
            gfa_dup = gfa.dup()
            for state in perm:
                eliminate_with_minimization(gfa_dup, state, delete_state=False)
            length = gfa_dup.delta[0][8].treeLength()
            min_length = min(min_length, length)
        length_list.append(min_length)
    with open('./result/length_list_' + str(n) + '.pkl', 'wb') as fp:
        dump(length_list, fp)

'''
for i in range(8, 51):
    print("i:", i)
    divide_and_conquer(i)

exit()
'''
'''
import statistics
with open('./result/length_list_1.pkl', 'rb') as fp:
    length_list = load(fp)

print('len: ', len(length_list))
print('avg: ', sum(length_list) / len(length_list))
print('std: ', statistics.stdev(length_list))
print('min: ', min(length_list))
print('max: ', max(length_list))
#'''

#import utils.random_position_automata_generator
