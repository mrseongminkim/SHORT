from FAdo.fio import *

import pickle

def load_data():
    data = []
    for n in range(3, 4):
        temp = []
        for d in ['_sparse', '_dense']:
            file_name = 'data/n' + str(n) + d
            try:
                with open(file_name + '.pickle',"rb") as fp:
                    content = pickle.load(fp)
            except:
                content = readFromFile(file_name + '.txt')
                with open(file_name + '.pickle',"wb") as fp:
                    pickle.dump(content, fp)
            temp.append(content)
        data.append(temp)
    return data