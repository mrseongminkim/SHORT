from FAdo.fio import *
from FAdo.conversions import *

from pickle import load, dump

def nfa_to_gfa(nfa: NFA):
    gfa = GFA()
    gfa.setSigma(nfa.Sigma)
    gfa.Initial = uSet(nfa.Initial)
    gfa.States = nfa.States[:]
    gfa.setFinal(nfa.Final)
    gfa.predecessors = {}
    for i in range(len(gfa.States)):
        gfa.predecessors[i] = set([])
    for s in nfa.delta:
        for c in nfa.delta[s]:
            for s1 in nfa.delta[s][c]:
                gfa.addTransition(s, c, s1)
    return gfa

#Change 4 to 11 when you done debugging
def load_data():
    data = []
    for n in range(3, 4):
        temp = []
        for d in ['_sparse', '_dense']:
            file_name = 'data/n' + str(n) + d
            try:
                with open(file_name + '.pickle',"rb") as fp:
                    content = load(fp)
            except:
                content = readFromFile(file_name + '.txt')
                for i in range(len(content)):
                    #content[i].display(d + str(i) + '.png')
                    #content[i] = nfa_to_gfa(content[i])
                    content[i].reorder({(content[i].States).index(x) : int(x) for x in content[i].States})
                #with open(file_name + '.pickle',"wb") as fp:
                #    dump(content, fp)
            temp.append(content)
        data.append(temp)
    return data