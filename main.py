import time
import csv

import numpy as np

from utils.data_loader import *
from utils.heuristics import *
from alpha_zero.utils import *
from alpha_zero.state_elimination.StateEliminationGame import StateEliminationGame as Game
from alpha_zero.state_elimination.pytorch.NNet import NNetWrapper as nn
from alpha_zero.MCTS import MCTS

args = dotdict({
    'numIters': 1000,
    'numEps': 100,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 40,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,
    'checkpoint': './alpha_zero/temp/',
    'load_model': True,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

'''
data = load_data()
exp = [[[[0, 0] for d in range(2)] for k in range(3)] for n in range(8)]

g = Game()
nnet = nn(g)
mcts = MCTS(g, nnet, args)
player = lambda x: np.argmax(mcts.getActionProb(x, temp=0))
curPlayer = 1

print(args.load_folder_file[0])
print(args.load_folder_file[1])

if args.load_model:
    nnet.load_checkpoint(args.checkpoint, args.load_folder_file[1])
else:
    print('sth is wrong')

for n in range(8):
    for k in range(3):
        for d in range(2):
            for i in range(100):
                print('n' + str(n + 3) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' NFA')
                gfa = data[n][k][d][i].dup()
                board = g.getInitBoard(gfa, n, k, d)
                start_time = time.time()
                while g.getGameEnded(board, curPlayer) == -1:
                    action = player(g.getCanonicalForm(board, curPlayer))
                    valids = g.getValidMoves(g.getCanonicalForm(board, curPlayer), 1)
                    if valids[action] == 0:
                        assert valids[action] > 0
                    board, curPlayer = g.getNextState(board, curPlayer, action)
                result = board[0][g.n + 1]
                end_time = time.time()
                result_time = end_time - start_time
                exp[n][k][d][0] += result_time
                exp[n][k][d][1] += result

with open('nn_experimental_result.pkl', 'wb') as fp:
    dump(exp, fp)
'''

exp = list()
with open('nn_experimental_result.pkl', 'rb') as fp:
    exp = load(fp)

with open('c7' + '.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for n in range(5 - 3, 11 - 3):
        size_value = exp[n][1][0][1] / 100
        writer.writerow([size_value])

exit()
'''
gfa = readFromFile('data.txt')
#gfa.display()
gfa = convert_nfa_to_gfa(gfa)
gfa.reorder({(gfa.States).index(x) : int(x) for x in gfa.States})

#result = decompose(gfa, False, False)
#print(result)
result = eliminate_by_repeated_state_weight_heuristic(gfa)
print(result)
exit()
#'''
'''
exp[0][n][k][d][0]: TIME = 0
exp[0][n][k][d][1]: LENGTH = 0
/ 20000
'''
'''
data = load_data()
exp = [[[[[0, 0] for d in range(2)] for k in range(3)] for n in range(8)] for c in range(6)]
for n in range(8):
    for k in range(3):
        for d in range(2):
            for i in range(100):
                print('n' + str(n + 3) + 'k' + ('2' if not k else ('5' if k == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' NFA')
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

with open('experimental_result.pkl', 'wb') as fp:
    dump(exp, fp)
#'''

'''
for c in range(6):
    with open('c' + str(c + 1) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for n in range(5 - 3, 11 - 3):
            size_value = exp[c][n][1][0][1] / 100
            writer.writerow([size_value])
exit()
'''

'''
for n in range(5 - 3, 11 - 3):
    with open('n' + str(n + 3) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for c in range(6):
            #time_value = exp[c][n][1][0][0] / 100
            size_value = exp[c][n][1][0][1] / 100
            #content = [time_value, size_value]
            writer.writerow([size_value])
'''


'''
exp = list()
with open('experimental_result.pkl', 'rb') as fp:
    exp = load(fp)

alpha = ['2', '5', '10']
density = ['0.2', '0.5']

for c in range(5, 6):
    with open('C_' + str(c + 1) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        title = ['x1', 'x2', 'x3', 'y']
        writer.writerow(title)
        for n in range(8):
            for k in range(3):
                for d in range(2):
                    #paramters = 'n = ' + str(n + 3) + ', k = ' + alpha[k] + ', d = ' + str(density[d])
                    size_value = exp[c][n][k][d][1] / 100
                    content = [str(n + 3), alpha[k], str(density[d]), size_value]
                    #content = [paramters, size_value]
                    writer.writerow(content)
'''