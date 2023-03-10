import matplotlib.pyplot as plt
import numpy as np
import csv

n = 6
r = np.arange(n)
width = 0.10
x = []

for c in range(1, 8):
    file = open("c" + str(c) + ".csv")
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
        label='C_6')

plt.bar(r + width * 6, x[6], color = 'black',
        width = width, edgecolor = 'black',
        label='C_7')

plt.xlabel("k = 5, d = 0.2")
plt.ylabel("Size of Regular Expressions")
plt.title("Average size of resulting regular expressions of state elimination")

plt.xticks(r + width/2,['n = 5','n = 6','n = 7','n = 8', 'n = 9', 'n = 10'])
plt.yscale('log', base=10)

plt.legend()
  
plt.show()