import time
import logging
import os
import random
import csv
import itertools
import sys
from pickle import load, dump

import torch
import numpy as np
import coloredlogs
from FAdo.conversions import *

from utils.data_loader import *
from utils.heuristics import *
from utils.random_nfa_generator import *

from alpha_zero.Coach import Coach
from alpha_zero.MCTS import MCTS
from alpha_zero.utils import *
from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame as Game
from alpha_zero.state_elimination.NNet import NNetWrapper as nn

from config import *

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')
args = dotdict({
    'numIters': 20,
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
    'cpuct': 3,
    'checkpoint': './alpha_zero/models/',
    'load_model': True,
    'load_folder_file': ('./alpha_zero/models/', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

def generate_test_data(type: str):
    if type == "nfa":
        generate_test_nfas()


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


def test_alpha_zero_without_mcts(model_updated, type, minimize):
    if not model_updated:
        with open("./result/rl_greedy_" + type + "_" + str(minimize) + ".pkl", "rb") as fp:
            exp = load(fp)
        with open("./result/rl_greedy_" + type + "_" + str(minimize) + ".csv", 'w', newline='') as fp:
            writer = csv.writer(fp)
            for i in range(N_RANGE):
                size_value = exp[i][0]
                time_value = exp[i][1]
                writer.writerow([size_value, time_value])
        return
    data = load_data(type)
    g = Game()
    nnet = nn(g)
    assert args.load_model
    nnet.load_checkpoint(args.checkpoint, args.load_folder_file[1])
    exp = [[0, 0] for i in range(N_RANGE)]
    for n in range(N_RANGE):
        for i in range(SAMPLE_SIZE):
            print('n:' + str(n + MIN_N) + ', i:', i)
            gfa = data[n][i]
            start_time = time.time()
            while g.getGameEnded(gfa) == None:
                board = g.gfaToBoard(gfa)
                policy, _ = nnet.predict(board)
                valid_moves = g.getValidMoves(gfa)
                policy = policy * valid_moves
                if not policy.any():
                    print("Poor prediction")
                    policy = valid_moves
                action = np.argmax(policy)
                eliminate_with_minimization(gfa, action, minimize=minimize)
            end_time = time.time()
            result_length = gfa.delta[0][1].treeLength()
            result_time = end_time - start_time
            exp[n][0] += result_length
            exp[n][1] += result_time
        exp[n][0] /= SAMPLE_SIZE
        exp[n][1] /= SAMPLE_SIZE
    with open("./result/rl_greedy_" + type + "_" + str(minimize) + ".pkl", "wb") as fp:
        dump(exp, fp)


'''
def test_alpha_zero(model_updated, type, n, minimize):
    model_updated = model_updated
    if not model_updated:
        if minimize:
            with open('./result/rl_' + str(n) + '_true.pkl', 'rb') as fp:
                exp = load(fp)
        else:
            with open('./result/rl_' + str(n) + '_false.pkl', 'rb') as fp:
                exp = load(fp)
        with open('./result/rl_' + str(n) + ('_true' if minimize else '_false') + '_length.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            size_value = exp[0] / sample_size
            writer.writerow([size_value])
        with open('./result/rl_' + str(n) + ('_true' if minimize else '_false') + '_time.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            time_value = exp[1] / sample_size
            writer.writerow([time_value])
    else:
        data = load_data(type)
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
        exp = [[0, 0] for i in range(8)]
        for n in range(8):
            for i in range(sample_size):
                mcts = MCTS(g, nnet, args)
                print('n: ' + str(n) + ', i:', i)
                gfa = data[n][i]
                gfa = g.getInitBoard(gfa, n + min_n, 5, 0.2)
                start_time = time.time()
                while g.getGameEnded(gfa, curPlayer) == None:
                    action = player(g.getCanonicalForm(gfa, curPlayer))
                    valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                    if valids[action] == 0:
                        assert valids[action] > 0
                    gfa, curPlayer = g.getNextState(gfa, curPlayer, action, minimize=minimize)
                end_time = time.time()
                result_length = gfa.delta[0][1].treeLength()
                result_time = end_time - start_time
                exp[n][0] += result_length
                exp[n][1] += result_time
        
        for i in range(8):
            print(exp[i][0] / sample_size)
        

        if minimize:
            with open('./result/rl_' + str(n) + '_true.pkl', 'wb') as fp:
                dump(exp, fp)
        else:
            with open('./result/rl_' + str(n) + '_false.pkl', 'wb') as fp:
                dump(exp, fp)
'''


def test_heuristics(model_updated, type, minimization):
    if not model_updated:
        with open("./result/heuristics_greedy_" + type + "_" + str(minimization) + ".pkl", "rb") as fp:
            exp = load(fp)
        for c in range(6):
            with open("./result/heuristics_greedy_" + type + "_" + str(minimization) + "_C" + str(c + 1) + ".csv", "w", newline="") as fp:
                writer = csv.writer(fp)
                for i in range(N_RANGE):
                    size_value = exp[c][i][0]
                    time_value = exp[c][i][1]
                    writer.writerow([size_value, time_value])
        return
    random.seed(SEED)
    data = load_data(type)
    exp = [[[0, 0] for n in range(N_RANGE)] for c in range(6)]
    for n in range(N_RANGE):
        for i in range(SAMPLE_SIZE):
            print('n: ' + str(n + MIN_N) + ', i:', i)
            CToken.clear_memory()
            gfa = data[n][i].dup()
            assert 0 not in gfa.delta[0]
            random_order = [i for i in range(1, len(gfa.States) - 1)]
            shuffle(random_order)
            decomposition_start_time = time.time()
            bridge_state_name = decompose(gfa)
            decomposition_end_time = time.time()
            decomposition_time = decomposition_end_time - decomposition_start_time

            # eliminate_randomly
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_randomly(gfa, minimization, random_order)
            end_time = time.time()
            result_time = end_time - start_time
            result_size = result.treeLength()
            exp[0][n][0] += result_size
            exp[0][n][1] += result_time
            c1_length = result_size
            c1_regex = result

            # decompose with eliminate_randomly
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_randomly(gfa, minimization, random_order, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[1][n][0] += result_size
            exp[1][n][1] += result_time
            c2_length = result_size
            c2_regex = result

            # eliminate_by_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_state_weight_heuristic(gfa, minimization)
            end_time = time.time()
            result_time = end_time - start_time
            result_size = result.treeLength()
            exp[2][n][0] += result_size
            exp[2][n][1] += result_time
            c3_length = result_size
            c3_regex = result

            # decompose + eliminate_by_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_state_weight_heuristic(gfa, minimization, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[3][n][0] += result_size
            exp[3][n][1] += result_time
            c4_length = result_size
            c4_regex = result

            # eliminate_by_repeated_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_repeated_state_weight_heuristic(gfa, minimization)
            end_time = time.time()
            result_time = end_time - start_time
            result_size = result.treeLength()
            exp[4][n][0] += result_size
            exp[4][n][1] += result_time
            c5_length = result_size
            c5_regex = result

            # decompose + eliminate_by_repeated_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_repeated_state_weight_heuristic(gfa, minimization, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[5][n][0] += result_size
            exp[5][n][1] += result_time
            c6_length = result_size
            c6_regex = result
            '''
            try:
                assert c1_length >= c2_length
                assert c3_length >= c4_length
                assert c5_length >= c6_length
            except:
                gfa = data[n][i].dup()
                print("States", gfa.States)
                print("Delta", gfa.delta)
                print("Initial", gfa.Initial)
                print("Final", gfa.Final)
                print("bridges", get_bridge_states(gfa))
                print("c1_length", c1_length)
                print("c2_lenght", c2_length)
                print("c3_length", c3_length)
                print("c4_lenght", c4_length)
                print("c5_length", c5_length)
                print("c6_lenght", c6_length)
                print("c1_regex", c1_regex)
                print("c2_regex", c2_regex)
                print("c3_regex", c3_regex)
                print("c4_regex", c4_regex)
                print("c5_regex", c5_regex)
                print("c6_regex", c6_regex)
                exit()
            '''
        for c in range(6):
            exp[c][n][0] /= SAMPLE_SIZE
            exp[c][n][1] /= SAMPLE_SIZE
    with open("./result/heuristics_greedy_" + type + "_" + str(minimization) + ".pkl", "wb") as fp:
        dump(exp, fp)


def test_optimal(model_updated, type, minimization):
    if not model_updated:
        with open("./result/optimal_" + type + "_" + str(minimization) + ".pkl", "rb") as fp:
            exp = load(fp)
        with open("./result/optimal_" + type + "_" + str(minimization) + ".csv", "w", newline="") as fp:
            writer = csv.writer(fp)
            for i in range(N_RANGE):
                size_value = exp[i]
                writer.writerow([size_value])
        return
    data = load_data(type)
    exp = [0 for n in range(N_RANGE)]
    for n in range(min(N_RANGE, 7)):
        for i in range(SAMPLE_SIZE):
            print('n: ' + str(n + MIN_N) + ', i:', i)
            CToken.clear_memory()
            gfa = data[n][i].dup()
            order = [i for i in range(1, len(gfa.States) - 1)]
            min_length = float("inf")
            for perm in itertools.permutations(order):
                result = eliminate_randomly(gfa, minimization, perm)
                min_length = min(min_length, result.treeLength())
                gfa = data[n][i].dup()
            exp[n] += min_length
        exp[n] /= SAMPLE_SIZE
    with open("./result/optimal_" + type + "_" + str(minimization) + ".pkl", "wb") as fp:
        dump(exp, fp)

'''
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
            mcts = MCTS(g, nnet, args)
            print("i:", i)
            gfa = data[i][0]
            original_length = data[i][1]
            rep_sw_length = eliminate_by_repeated_state_weight_heuristic(gfa.dup()).treeLength()
            rand_length = eliminate_randomly(gfa.dup()).treeLength()
            gfa = g.getInitBoard(gfa, len(gfa.States) - 2)
            while g.getGameEnded(gfa, curPlayer) == None:
                action = player(g.getCanonicalForm(gfa, curPlayer))
                valids = g.getValidMoves(g.getCanonicalForm(gfa, curPlayer), curPlayer)
                if valids[action] == 0:
                    assert valids[action] > 0
                gfa, curPlayer = g.getNextState(gfa, curPlayer, action)
            exp += gfa.delta[0][1].treeLength()
            original += original_length
            print('original:', original_length)
            print('lr:', gfa.delta[0][1].treeLength())
            print('c6:', rep_sw_length)
            print('random:', rand_length)
        exp /= sample_size
        original /= sample_size
        with open('./result/alpha_zero_position_result.pkl', 'wb') as fp:
            dump(exp, fp)
        with open('./result/postion_original_length.pkl', 'wb') as fp:
            dump(original, fp)


def test_brute_force(model_updated, type):
    model_updated = model_updated
    if not model_updated and os.path.isfile('./result/brute_force_experiment_result.pkl'):
        with open('./result/brute_force_experiment_result_minimize_10.pkl', 'rb') as fp:
            exp = load(fp)
        with open('./result/optimal_length_minimize_10.csv', 'w', newline='') as fp:
            writer = csv.writer(fp)
            for n in range(5):
                size_value = exp[n] / sample_size
                writer.writerow([size_value])
    else:
        data = load_data(type)
        exp = [0] * 5
        for n in range(3 - 3, 8 - 3):
            for i in range(sample_size):
                permutations = [x for x in range(1, len(data[n][i].States) - 1)]
                print('n' + str(n + min_n) + '\'s' + str(i + 1) + 'sample')
                min_length = float('inf')
                start_time = time.time()
                for perm in itertools.permutations(permutations):
                    gfa = data[n][i].dup()
                    for state in perm:
                        eliminate_with_minimization(gfa, state, delete_state=False, minimize=True)
                    if gfa.Initial in gfa.delta and gfa.Initial in gfa.delta[gfa.Initial]:
                        length = CConcat(CStar(gfa.delta[gfa.Initial][gfa.Initial]), gfa.delta[gfa.Initial][list(gfa.Final)[0]]).treeLength()
                        print('This will never run, since i specifically made GFA to not have returnining transition')
                    else:
                        length = gfa.delta[gfa.Initial][list(gfa.Final)[0]].treeLength()
                    min_length = min(min_length, length)
                exp[n] += min_length
            print(exp[n] / min_length)
        with open('./result/brute_force_experiment_result_minimize_10.pkl', 'wb') as fp:
            dump(exp, fp)


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
    if sys.argv[1] == "train":
        train_alpha_zero()
        exit()
    
    if sys.argv[2] == 'DFA':
        type = 'dfa'
    elif sys.argv[2] == 'NFA':
        type = 'nfa'
    else:
        print("Specify the target FAs")
        exit()
    if sys.argv[3] == 'true':
        minimization = True
    elif sys.argv[3] == 'false':
        minimization = False
    else:
        print("Specify whether enable or disable minimization")
        exit()
    
    if sys.argv[1] == 'rl':
        for n in range(3, 11):
            test_alpha_zero(True, type, n, minimization)
            test_alpha_zero(False, type, n, minimization)
    elif sys.argv[1] == 'heuristics':
        test_heuristics(True, type, minimization)
        test_heuristics(False, type, minimization)

#main()
'''
#test_alpha_zero_without_mcts(True, "nfa", True)
#test_alpha_zero_without_mcts(False, "nfa", True)

#generate_test_data("nfa")

#train_alpha_zero()

#generate_test_data("nfa")
#test_heuristics(model_updated=True, type="nfa", minimization=False)
#test_heuristics(model_updated=False, type="nfa", minimization=False)

test_optimal(model_updated=True, type="nfa", minimization=False)
test_optimal(model_updated=False, type="nfa", minimization=False)

#generate_test_data("nfa")