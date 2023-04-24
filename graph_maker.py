import matplotlib.pyplot as plt
import numpy as np
import csv

def primitive():
    type = "length"
    n = 8
    r = np.arange(n)
    width = 0.10
    x = []
    
    for i in range(1, 7):
        file = open("./result/true/c" + str(i) + "_" + type + ".csv")
        temp = []
        for row in csv.reader(file):
            temp.append(row[0])
        temp = [float(i) for i in temp]
        x.append(temp)
        file.close()
    
    file = open("./result/rl" + "_" + type + ".csv")
    temp = []
    for row in csv.reader(file):
        temp.append(row[0])
    temp = [float(i) for i in temp]
    x.append(temp)
    file.close()

    color = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:cyan', 'tab:olive']
    for i in range(0, 6):
        plt.bar(r + width * i, x[i], color=color[i], width=width, edgecolor='black', label='C-' + str(i + 1))

    plt.bar(r + width * 6, x[6], color='k', width=width, edgecolor='black', label='Ours')

    plt.xlabel("k = 5, d = 0.2")
    plt.ylabel("Size of Regular Expressions")
    plt.title("Average size of resulting regular expressions of state elimination from random NFAs")

    plt.xticks(r + width / 2, ['n = 3','n = 4','n = 5','n = 6', 'n = 7', 'n = 8', 'n = 9', 'n = 10'])
    plt.yscale('log', base=10)

    plt.legend()
    
    plt.show()


def simplification():
    type = "length"
    n = 8
    r = np.arange(n)
    width = 0.1
    x = []
    
    for i in range(1, 7):
        file = open("./result/false/c" + str(i) + "_" + type + ".csv")
        temp = []
        for row in csv.reader(file):
            temp.append(row[0])
        big = [float(i) for i in temp]
        #x.append(temp)
        file.close()

        file = open("./result/true/c" + str(i) + "_" + type + ".csv")
        temp = []
        for row in csv.reader(file):
            temp.append(row[0])
        small = [float(i) for i in temp]
        #x.append(temp)
        file.close()

        temp = []
        for i in range(len(big)):
            temp.append(big[i] - small[i])
        x.append(temp)

    color = ['tab:red', 'tab:green', 'tab:blue', 'tab:purple', 'tab:brown', 'tab:orange']
    for i in range(0, 6):
        plt.bar(r + width * i, x[i], color=color[i], width=width, edgecolor='black', label='C-' + str(i + 1))

    plt.xlabel("k = 5, d = 0.2")
    plt.ylabel("Size of Simplified Regular Expressions")
    plt.title("Average size of simplified resulting regular expressions of state elimination from random NFAs")

    plt.xticks(r + width / 2, ['n = 3','n = 4','n = 5','n = 6', 'n = 7', 'n = 8', 'n = 9', 'n = 10'])
    plt.yscale('log', base=10)

    plt.legend()
    
    plt.show()


#simplification()

primitive()