import matplotlib.pyplot as plt
import numpy as np
import csv

def token():
    token_enabled = [0.000180101, 0.000459738, 0.001460278, 0.004479399, 0.009266975, 0.029643726, 0.109175293, 0.380451472]
    token_disabled = [0.000181348, 0.000830235, 0.004292531, 0.023661940, 0.125375790, 0.581623246, 2.607046785, 12.61751385]
    x = range(3, 11)
    plt.title("Average Time for State Elimination in Random Ordering on Random NFAs")
    plt.plot(x, token_enabled, label='CToken Enabled')
    plt.plot(x, token_disabled, label='CToken Disabled')
    plt.legend()
    plt.xlabel('number of states')
    plt.ylabel('Elapsed time')
    #plt.show()
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.savefig('CToken.pdf', bbox_inches='tight')
    #plt.show()

    '''
    n = 8
    r = np.arange(n)
    width = 0.10
    x = []
    x.append(token_enabled)
    x.append(token_disabled)
    color = ['tab:red', 'tab:blue']
    label = ['symboling enabled', 'symboling disabled']
    for i in range(2):
        plt.bar(r + width * i, x[i], color=color[i], width=width, edgecolor='black', label=label[i])
    plt.xlabel("k = 5, d = 0.2")
    plt.ylabel("Time elapsed")
    plt.title("Average time to finish randomly order state elmination from random NFAs")
    plt.xticks(r + width / 2, ['n = 3','n = 4','n = 5','n = 6', 'n = 7', 'n = 8', 'n = 9', 'n = 10'])
    #plt.yscale('log', base=10)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(6, 3)
    fig.savefig('symboling.pdf', bbox_inches='tight')
    #plt.show()

    # Create the plot
    plt.plot(x, token_enabled, label='Token Enabled')
    plt.plot(x, token_disabled, label='Token Disabled')

    # Set up the legend and labels
    plt.legend()
    plt.xlabel('Data Points')
    plt.ylabel('Value')

    # Show the plot
    plt.show()
    '''

token()
exit()

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