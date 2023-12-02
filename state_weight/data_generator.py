import sys
sys.path.append(r"/home/toc/seongmin")
sys.path.append(r"/home/toc/seongmin/SHORT")

import csv
import pickle
import tqdm

from FAdo.conversions import GFA

from StateEliminationGame import StateEliminationGame
from SHORT.utils.fadomata import get_weight

game = StateEliminationGame(maxN=5)

annotation_file = open("./state_weight_data/annotations_file.csv", "w", newline="")
writer = csv.writer(annotation_file)
count = 0
data_size = 100_000 #80_000 for train and 20_000 for valid
progress = tqdm.tqdm(total=data_size)

while count != data_size:
    gfa: GFA = game.get_initial_gfa(n=5, k=5, d=0.1)
    if game.getGameEnded(gfa):
        continue
    step = 1
    while game.getGameEnded(gfa) == None:
        state_weight = [0] * game.getActionSize()
        min_weight = float("inf")
        min_idx = -1
        for i in range(len(gfa.States)):
            weight = get_weight(gfa, i)
            state_weight[i] = weight
            if i != gfa.Initial and i not in gfa.Final and weight < min_weight:
                min_weight = weight
                min_idx = i
        with open(f"./state_weight_data/gfa{count}_step{step}.pkl", "wb") as fp:
            pickle.dump(gfa, fp)
        writer.writerow([f"gfa{count}_step{step}.pkl", state_weight])
        step += 1
        gfa = game.getNextState(gfa, min_idx)
    count += 1
    progress.update(1)
