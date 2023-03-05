import time
import logging
import os
import random

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
coloredlogs.install(level = 'INFO')
args = dotdict({
    'numIters' : 1000,
    'numEps' : 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold' : 15,        # temperature hyperparameters
    'updateThreshold' : 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue' : 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims' : 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare' : 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct' : 1,
    'checkpoint' : './alpha_zero/models/',
    'load_model' : False,
    'load_folder_file' : ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory' : 20,
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
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
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
    if os.path.isfile('./result/alpha_zero_experiment_result.pkl'):
        with open('./result/alpha_zero_experiment_result.pkl', 'rb') as fp:
            exp = load(fp)
    else:
        data = load_data()
        exp = [[[[0, 0] for d in range(len(density))] for k in range(len(alphabet))] for n in range(n_range)]
        g = Game()
        nnet = nn(g)
        mcts = MCTS(g, nnet, args)
        player = lambda x: np.argmax(mcts.getActionProb(x, temp = 0))
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
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        gfa = data[n][k][d][i].dup()
                        board = g.getInitBoard(gfa, n + min_n, alphabet[k], density[d])
                        order = []
                        start_time = time.time()
                        while g.getGameEnded(board, curPlayer) == -1:
                            action = player(g.getCanonicalForm(board, curPlayer))
                            valids = g.getValidMoves(g.getCanonicalForm(board, curPlayer), 1)
                            if valids[action] == 0:
                                assert valids[action] > 0
                            board, curPlayer = g.getNextState(board, curPlayer, action)
                            order.append(action + 1)
                        result = board[0][n + min_n + 1]
                        end_time = time.time()
                        gfa.eliminateAll(order)
                        if (result != gfa.delta[0][n + min_n + 1].treeLength()):
                            print('order', order)
                            print('result length', result)
                            print('valid length', gfa.delta[0][n + min_n + 1].treeLength())
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
        exp = [[[[[0, 0] for d in range(len(density))] for k in range(len(alphabet))] for n in range(n_range)] for c in range(6)]
        for n in range(n_range):
            for k in range(len(alphabet)):
                for d in range(len(density)):
                    for i in range(sample_size):
                        random.seed(i)
                        print('n' + str(n + min_n) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' sample')
                        #eliminate_randomly
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_randomly(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[0][n][k][d][0] += result_time
                        exp[0][n][k][d][1] += result_size

                        #decompose with eliminate_randomly
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = decompose(gfa, False, False)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[1][n][k][d][0] += result_time
                        exp[1][n][k][d][1] += result_size

                        #eliminate_by_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_by_state_weight_heuristic(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[2][n][k][d][0] += result_time
                        exp[2][n][k][d][1] += result_size

                        #decompose + eliminate_by_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = decompose(gfa, True, False)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[3][n][k][d][0] += result_time
                        exp[3][n][k][d][1] += result_size

                        #eliminate_by_repeated_state_weight_heuristic
                        gfa = data[n][k][d][i].dup()
                        start_time = time.time()
                        result = eliminate_by_repeated_state_weight_heuristic(gfa)
                        end_time = time.time()
                        result_time = end_time - start_time
                        result_size = result.treeLength()
                        exp[4][n][k][d][0] += result_time
                        exp[4][n][k][d][1] += result_size

                        #decompose + eliminate_by_repeated_state_weight_heuristic
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

def main():
    train_alpha_zero()

main()