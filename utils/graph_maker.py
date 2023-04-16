import matplotlib.pyplot as plt
import numpy as np
import csv

n = 5
r = np.arange(n)
width = 0.10
x = []
#'''
file = open("./result/c6_reduction.csv")
temp = []
for row in csv.reader(file):
    temp.append(row[0])
temp = [float(i) for i in temp]
x.append(temp)
file.close()

file = open("./result/c8.csv")
temp = []
for row in csv.reader(file):
    temp.append(row[0])
temp = [float(i) for i in temp]
x.append(temp)
file.close()

file = open("./result/maxn50_iter18_mcts25.csv")
temp = []
for row in csv.reader(file):
    temp.append(row[0])
temp = [float(i) for i in temp]
x.append(temp)
file.close()

file = open("./result/maxn50_iter18_mcts50.csv")
temp = []
for row in csv.reader(file):
    temp.append(row[0])
temp = [float(i) for i in temp]
x.append(temp)
file.close()

file = open("./result/maxn50_iter18_mcts100.csv")
temp = []
for row in csv.reader(file):
    temp.append(row[0])
temp = [float(i) for i in temp]
x.append(temp)
file.close()

plt.bar(r, x[0], color = 'r',
        width = width, edgecolor = 'black',
        label='C6 with reduction')

plt.bar(r + width, x[1], color = 'g',
        width = width, edgecolor = 'black',
        label='Optimal without reduction')

plt.bar(r + width * 2, x[2], color = 'b',
        width = width, edgecolor = 'black',
        label='mcts25')

plt.bar(r + width * 3, x[3], color = 'm',
        width = width, edgecolor = 'black',
        label='mcts50')

plt.bar(r + width * 4, x[4], color = 'c',
        width = width, edgecolor = 'black',
        label='mcts100')
#'''
'''
for c in range(1, 9):
    file = open("./result/c" + str(c) + ".csv")
    temp = []
    for row in csv.reader(file):
        temp.append(row[0])
    temp = [float(i) for i in temp]
    x.append(temp)

plt.bar(r, x[0], color = 'r',
        width = width, edgecolor = 'black',
        label='C_1')

plt.bar(r + width, x[1], color = 'g',
        width = width, edgecolor = 'black',
        label='C_2')

plt.bar(r + width * 2, x[2], color = 'b',
        width = width, edgecolor = 'black',
        label='C_3')

plt.bar(r + width * 3, x[3], color = 'm',
        width = width, edgecolor = 'black',
        label='C_4')

plt.bar(r + width * 4, x[4], color = 'c',
        width = width, edgecolor = 'black',
        label='C_5')

plt.bar(r + width * 5, x[5], color = 'y',
        width = width, edgecolor = 'black',
        label='C6 without reduction')

plt.bar(r + width * 6, x[6], color = 'black',
        width = width, edgecolor = 'black',
        label='alpha zero with reduction')

plt.bar(r + width * 7, x[7], color = 'white',
        width = width, edgecolor = 'black',
        label='optimal')
'''

plt.xlabel("k = 5, d = 0.2")
plt.ylabel("Size of Regular Expressions")
plt.title("Average size of resulting regular expressions of state elimination")

plt.xticks(r + width / 2, ['n = 3','n = 4','n = 5','n = 6', 'n = 7'])
plt.yscale('log', base=10)

plt.legend()
  
plt.show()