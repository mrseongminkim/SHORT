import time

from utils.data_loader import *
from utils.heuristics import *

'''
exp[0][n][k][d][0]: TIME = 0
exp[0][n][k][d][1]: LENGTH = 0
/ 20000
'''
data = load_data()
'''
exp = [[[[0, 0] for d in range(2)] for k in range(3)] for n in range(8)]

for n in range(8):
    for k in range(3):
        for d in range(2):
            for i in range(20000):
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
'''