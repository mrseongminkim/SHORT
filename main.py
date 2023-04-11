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

from test import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
args = dotdict({
    'numIters': 1000,
    # Number of complete self-play games to simulate during a new iteration.
    'numEps': 100,
    #'tempThreshold': 4,        # temperature hyperparameters
    # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'updateThreshold': 0.6,
    # Number of game examples to train the neural networks.
    'maxlenOfQueue': 200000,
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    # Number of games to play during arena play to determine if new net will be accepted.
    'arenaCompare': 40,
    'cpuct': 1,
    'checkpoint': './alpha_zero/models/deleting/',
    'load_model': False,
    #'load_folder_file': ('./alpha_zero/models/', 'best.pth.tar'),
    'load_folder_file': ('./alpha_zero/models/deleting/', 'checkpoint_1.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})
min_n = 3
max_n = 10
n_range = max_n - min_n + 1
alphabet = [2, 5, 10]
density = [0.2, 0.5]
sample_size = 100

def train_alpha_zero():
    print("Let's briefly check the important hyperparameters.")
    print("\tnumMCTSSims: ", args.numMCTSSims)
    print("\tcuda: ", torch.cuda.is_available())
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


def test_alpha_zero(model_updated):
    model_updated = model_updated
    if not model_updated and os.path.isfile('./result/alpha_zero_experiment_result.pkl'):
        with open('./result/alpha_zero_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/c7_iter_10_4.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5):
                #k = 5, d = 0.2
                size_value = exp[n][1][0][1] / 100
                writer.writerow([size_value])
    else:
        data = load_data('nfa')
        exp = [[[[0, 0] for d in range(len(density))] for k in range(len(alphabet))] for n in range(n_range)]
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
                        if d == 1 or k != 1 or n > 4:
                            continue
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        gfa = data[n][k][d][i]
                        gfa = g.getInitBoard(gfa, n + min_n, alphabet[k], density[d])
                        start_time = time.time()
                        while g.getGameEnded(gfa, curPlayer) == -1:
                            action = player(g.getCanonicalForm(gfa, curPlayer))
                            valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                            if valids[action] == 0:
                                assert valids[action] > 0
                            gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
                        end_time = time.time()
                        result_length = gfa.delta[0][1].treeLength()
                        result_time = end_time - start_time
                        exp[n][k][d][0] += result_time
                        exp[n][k][d][1] += result_length
        with open('./result/alpha_zero_experiment_result.pkl', 'wb') as fp:
            dump(exp, fp)


def test_heuristics():
    model_updated = False
    if not model_updated and os.path.isfile('./result/heuristics_experiment_result.pkl'):
        with open('./result/heuristics_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
        for c in range(6):
            with open('./result/c' + str(c + 1) + '.csv', 'w', newline='') as fp:
                writer = csv.writer(fp)
                for n in range(5):
                    size_value = exp[c][n][1][0][1] / 100
                    writer.writerow([size_value])
    else:
        data = load_data()
        exp = [[[[[0, 0] for d in range(len(density))] for k in range(len(alphabet))] for n in range(n_range)] for c in range(6)]
        for n in range(n_range):
            for k in range(len(alphabet)):
                for d in range(len(density)):
                    for i in range(sample_size):
                        if d == 1 or k != 1 or n > 4:
                            continue
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        '''
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
                        result = eliminate_by_repeated_state_weight_heuristic(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[4][n][k][d][0] += result_time
                        exp[4][n][k][d][1] += result_size
                        '''

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


def test_alpha_zero_for_position():
    model_updated = False
    if not model_updated and os.path.isfile('./result/alpha_zero_position_result.pkl'):
        with open('./result/alpha_zero_position_result.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/postion_original_length.pkl', 'rb') as fp:
            average_origianl_length = load(fp)
        with open('./result/position.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5):
                size_value = exp[n]
                writer.writerow([size_value])
        with open('./result/original_position.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5):
                size_value = average_origianl_length[n]
                writer.writerow([size_value])
    else:
        data = load_data('position')
        exp = [0] * 5
        average_origianl_length = [0] * 5
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
        for n in range(5):
            for i in range(100):
                print('n', n, 'i', i)
                gfa = data[n][i][0]
                original_length = data[n][i][1]
                gfa = g.getInitBoard(gfa, len(gfa.States) - 2)
                while g.getGameEnded(gfa, curPlayer) == -1:
                    action = player(g.getCanonicalForm(gfa, curPlayer))
                    valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                    if valids[action] == 0:
                        assert valids[action] > 0
                    gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
                result_length = gfa.delta[0][1].treeLength()
                exp[n] += result_length
                average_origianl_length[n] += original_length
            exp[n] /= 100
            average_origianl_length[n] /= 100
        with open('./result/alpha_zero_position_result.pkl', 'wb') as fp:
            dump(exp, fp)
        with open('./result/postion_original_length.pkl', 'wb') as fp:
            dump(average_origianl_length, fp)

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
    while g.getGameEnded(gfa, curPlayer) == -1:
        action = player(g.getCanonicalForm(gfa, curPlayer))
        valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
        if valids[action] == 0:
            assert valids[action] > 0
        order.append(gfa.States[action])
        gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
    print(order)

def main():
    print("deleting-states")
    train_alpha_zero()
    #test_alpha_zero(True)
    #test_alpha_zero(False)

main()
