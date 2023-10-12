import time
import logging
import random
import csv
import itertools
from pickle import load, dump
from statistics import mean, stdev

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

torch.set_printoptions(precision=4, sci_mode=False, linewidth=512)
np.set_printoptions(precision=4, linewidth=512, suppress=True)

log = logging.getLogger(__name__)
coloredlogs.install(level='INFO')

def generate_test_data(type: str):
    if type == "nfa":
        generate_test_nfas()

def train_alpha_zero():
    log.info('Loading %s...', Game.__name__)
    g = Game()
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    if LOAD_MODEL:
        log.info('Loading checkpoint "%s/%s"...', LOAD_FOLDER_FILE[0], LOAD_FOLDER_FILE[1])
        nnet.load_checkpoint(LOAD_FOLDER_FILE[0], LOAD_FOLDER_FILE[1])
    else:
        log.warning('Not loading a checkpoint!')
    log.info('Loading the Coach...')
    c = Coach(g, nnet)
    if LOAD_MODEL:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()
    #c.load_initial_data()
    log.info('Starting the learning process')
    c.learn()

def single_data_for_train_alpha_zero():
    log.info('Loading %s...', Game.__name__)
    g = Game()
    log.info('Loading %s...', nn.__name__)
    nnet = nn(g)
    log.info('Loading the Coach...')
    c = Coach(g, nnet)
    log.info('Starting the learning process')
    gfa: GFA = g.get_initial_gfa()
    gfa_original = gfa.dup()
    mcts = MCTS(g, nnet)
    train_gfa = g.gfa_to_tensor(gfa)
    train_pi = None
    train_v = None
    while g.getGameEnded(gfa) == None:
        pi = mcts.getActionProb(gfa)
        if train_pi == None:
            train_pi = pi
        best_actions = np.array(np.argwhere(pi == np.max(pi))).flatten()
        best_action = np.random.choice(best_actions)
        best_pi = [0] * len(pi)
        best_pi[best_action] = 1
        action = np.random.choice(len(best_pi), p=best_pi)
        gfa = g.getNextState(gfa, action)
        #print(gfa.delta)
    train_v = -g.getGameEnded(gfa)
    train_data = [[train_gfa, train_pi, train_v]]

    examplesFile = os.path.join(LOAD_FOLDER_FILE[0], "libera_me.pkl")
    with open(examplesFile, "wb") as f:
        dump((gfa_original, train_data), f)
    
    print(train_pi)
    #nnet.train(train_data)

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
    assert LOAD_MODEL
    nnet.load_checkpoint(CHECKPOINT, LOAD_FOLDER_FILE[1])
    exp = [[0, 0] for i in range(N_RANGE)]
    for n in range(N_RANGE):
        for i in range(SAMPLE_SIZE):
            #if n + MIN_N != VICTIM: continue
            print('n:' + str(n + MIN_N) + ', i:', i)
            CToken.clear_memory()
            gfa = data[n][i]
            start_time = time.time()
            while g.getGameEnded(gfa) == None:
                gfa_representation = g.gfa_to_tensor(gfa)
                policy, _ = nnet.predict(gfa_representation)
                #print("mcts off\t:", policy[:10])
                #return
                valid_moves = g.getValidMoves(gfa)
                policy = policy * valid_moves
                if not policy.any():
                    print("Poor prediction")
                    policy = valid_moves
                action = np.argmax(policy)
                eliminate_with_minimization(gfa, action, minimize=minimize)
            end_time = time.time()
            result = g.get_resulting_regex(gfa)
            result_length = result.treeLength()
            #print("result length:", result_length)
            result_time = end_time - start_time
            exp[n][0] += result_length
            exp[n][1] += result_time
        exp[n][0] /= SAMPLE_SIZE
        exp[n][1] /= SAMPLE_SIZE
    with open("./result/rl_greedy_" + type + "_" + str(minimize) + ".pkl", "wb") as fp:
        dump(exp, fp)

def test_alpha_zero_with_mcts(model_updated, type, minimize):
    if not model_updated:
        with open("./result/rl_search_" + type + "_" + str(minimize) + ".pkl", "rb") as fp:
            exp = load(fp)
        with open("./result/rl_search_" + type + "_" + str(minimize) + ".csv", 'w', newline='') as fp:
            writer = csv.writer(fp)
            for i in range(N_RANGE):
                size_value = exp[i][0]
                time_value = exp[i][1]
                writer.writerow([size_value, time_value])
        return
    data = load_data(type)
    g = Game()
    nnet = nn(g)
    assert LOAD_MODEL
    nnet.load_checkpoint(CHECKPOINT, LOAD_FOLDER_FILE[1])
    exp = [[0, 0] for i in range(N_RANGE)]
    for n in range(N_RANGE):
        for i in range(SAMPLE_SIZE):
            if n + MIN_N != VICTIM: continue
            print('n:' + str(n + MIN_N) + ', i:', i)
            CToken.clear_memory()
            mcts = MCTS(g, nnet)
            gfa = g.get_initial_gfa(data[n][i], n + MIN_N, 5, 0.1)
            start_time = time.time()
            while g.getGameEnded(gfa) == None:
                pi = mcts.getActionProb(gfa)
                pi = np.array(pi)
                print("mcts on \t:", pi[:10])
                #return
                best_actions = np.array(np.argwhere(pi == np.max(pi))).flatten()
                best_action = np.random.choice(best_actions)
                best_pi = [0] * len(pi)
                best_pi[best_action] = 1
                action = np.random.choice(len(best_pi), p=best_pi)
                gfa = g.getNextState(gfa, action)
            end_time = time.time()
            result = g.get_resulting_regex(gfa)
            result_length = result.treeLength()
            #print("dead_end:", len(mcts.dead_end))
            print("result length:", result_length)
            result_time = end_time - start_time
            exp[n][0] += result_length
            exp[n][1] += result_time
        exp[n][0] /= SAMPLE_SIZE
        exp[n][1] /= SAMPLE_SIZE
    with open("./result/rl_search_" + type + "_" + str(minimize) + ".pkl", "wb") as fp:
        dump(exp, fp)

def get_sample_distribution(model_updated, minimization=False):
    if not model_updated:
        with open("./result/distribution.pkl", "rb") as fp:
            exp = load(fp)
        with open("./result/distribution.csv", "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([exp[0], exp[1]])
        return
    lst = []
    game = Game()
    for i in range(SAMPLE_SIZE):
        print("i:", i)
        CToken.clear_memory()
        gfa = game.get_initial_gfa()
        order = [i for i in range(1, len(gfa.States) - 1)]
        min_length = float("inf")
        for perm in itertools.permutations(order):
            result = eliminate_randomly(gfa.dup(), minimization, perm)
            min_length = min(min_length, result.treeLength())
        lst.append(min_length)
    avg = mean(lst)
    std = stdev(lst)
    exp = [avg, std]
    with open("./result/distribution.pkl", "wb") as fp:
        dump(exp, fp)

def get_optimal_ordering(minimization=False):
    data = load_data("nfa")
    for n in range(N_RANGE):
        length = 0
        for i in range(SAMPLE_SIZE):
            #if n + MIN_N != 6: continue
            print(f"n: {n}, i: {i}")
            CToken.clear_memory()
            gfa = data[n][i]
            order = [i for i in range(len(gfa.States)) if i != gfa.Initial and i not in gfa.Final]
            #print("order:", order)
            min_length = float("inf")
            #optimal_ordering = []
            for perm in itertools.permutations(order):
                result = eliminate_randomly(gfa.dup(), minimization, perm)
                #print("perm:", [gfa.States[x] for x in perm])
                #print("length:", result.treeLength())
                if min_length > result.treeLength():
                    min_length = result.treeLength()
                    #optimal_ordering = perm
            #optimal_ordering = [gfa.States[x] for x in optimal_ordering]
            #print("min_length:", min_length)
            #print("optimal_ordering", optimal_ordering)
            length += min_length
        print(f"{n} opt length:", length / SAMPLE_SIZE)

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
            random_order = [i for i in range(len(gfa.States)) if i != gfa.Initial and i not in gfa.Final]
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

            # decompose with eliminate_randomly
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_randomly(gfa, minimization, random_order, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[1][n][0] += result_size
            exp[1][n][1] += result_time

            # eliminate_by_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_state_weight_heuristic(gfa, minimization)
            end_time = time.time()
            result_time = end_time - start_time
            result_size = result.treeLength()
            exp[2][n][0] += result_size
            exp[2][n][1] += result_time

            # decompose + eliminate_by_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_state_weight_heuristic(gfa, minimization, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[3][n][0] += result_size
            exp[3][n][1] += result_time

            # eliminate_by_repeated_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_repeated_state_weight_heuristic(gfa, minimization)
            end_time = time.time()
            result_time = end_time - start_time
            result_size = result.treeLength()
            exp[4][n][0] += result_size
            exp[4][n][1] += result_time

            # decompose + eliminate_by_repeated_state_weight_heuristic
            gfa = data[n][i].dup()
            start_time = time.time()
            result = eliminate_by_repeated_state_weight_heuristic(gfa, minimization, bridge_state_name)
            end_time = time.time()
            result_time = end_time - start_time + decomposition_time
            result_size = result.treeLength()
            exp[5][n][0] += result_size
            exp[5][n][1] += result_time

        for c in range(6):
            exp[c][n][0] /= SAMPLE_SIZE
            exp[c][n][1] /= SAMPLE_SIZE
    with open("./result/heuristics_greedy_" + type + "_" + str(minimization) + ".pkl", "wb") as fp:
        dump(exp, fp)

def libera_me():
    examplesFile = os.path.join(LOAD_FOLDER_FILE[0], "libera_me.pkl")
    with open(examplesFile, "rb") as f:
        gfa, train_data = load(f)
    graph, y_pi, y_v = train_data[0]
    y_pi = np.array(y_pi)

    print("States:", gfa.States)
    print("delta:", gfa.delta)

    g = Game()
    nnet = nn(g)
    assert LOAD_MODEL
    nnet.load_checkpoint(CHECKPOINT, LOAD_FOLDER_FILE[1])

    pi, v = nnet.predict(graph)

    '''
    print("pi:", pi[:8])
    print("y_pi:", y_pi[:8])
    print("train_pi", v)
    print("y_v", y_v)
    '''

train_alpha_zero()
