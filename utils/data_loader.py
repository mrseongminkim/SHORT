from pickle import load, dump

from FAdo.fio import *

from fadomata import *

def load_data() -> list:
    data = []
    for n in range(3, 11):
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