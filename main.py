import time
import csv

from utils.data_loader import *
from utils.heuristics import *

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
                print('n' + str(n + 3) + 'k' + ('2' if not d else ('5' if d == 1 else '10')) + ('s' if not d else 'd') + '\'s ' + str(i + 1) + ' NFA')
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

exp = list()
with open('experimental_result.pkl', 'rb') as fp:
    exp = load(fp)

alpha = ['2', '5', '10']
density = ['0.2', '0.5']

for c in range(6):
    with open('C_' + str(c + 1) + '.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        title = ['n, k, d', 'time', 'size']
        writer.writerow(title)
        for n in range(8):
            for k in range(3):
                for d in range(2):
                    paramters = 'n = ', str(n + 3) + ', k = ' + alpha[k] + ', d = ' + density[d]
                    time_value = exp[c][n][k][d][0] / 100
                    size_value = exp[c][n][k][d][1] / 100
                    content = [paramters, time_value, size_value]
                    writer.writerow(content)