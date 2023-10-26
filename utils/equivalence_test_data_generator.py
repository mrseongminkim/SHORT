import csv
import pickle

from FAdo.fa import NFA
from FAdo.conversions import GFA

from utils.random_nfa_generator import generate
from utils.fadomata import convert_nfa_to_gfa, shuffle_fa

annotation_file = open("./equivalence_test_data/annotations_file.csv", "w", newline="")
writer = csv.writer(annotation_file)

NUM_OF_ORDER = 10
NUM_OF_ELIM = 10

#Total: 1,350,000
#Positive case: 675,000
#Negative case: 675,000

#there will be 2 * 90 * number_of_pairs of data cases (accept / reject)
#there will be 2 * 10 * number_of_pairs NFAs
#number_of_pairs = 556
#count = 0
for _ in range(number_of_pairs):
    nfa_1: NFA = generate(10, 5, 0.1, "nfa")
    nfa_2: NFA = generate(10, 5, 0.1, "nfa")
    nfa_1.Sigma = {'0', '1', '2', '3', '4', '5'}
    nfa_2.Sigma = {'0', '1', '2', '3', '4', '5'}
    while nfa_1 == nfa_2:
        nfa_2: NFA = generate(10, 5, 0.1, "nfa")
        nfa_2.Sigma = {'0', '1', '2', '3', '4', '5'}
    nfa_1: GFA = convert_nfa_to_gfa(nfa_1)
    nfa_2: GFA = convert_nfa_to_gfa(nfa_2)
    #now nfa_1 and nfa_2 are different

    nfa_1_equivalences = [nfa_1.dup()]
    for __ in range(NUM_OF_ORDER):
        nfa_1_equivalences.append(shuffle_fa(nfa_1.dup()))
    for i in range(NUM_OF_ELIM):
        nfa_1.eliminateState(1)
        nfa_1_equivalences.append(nfa_1)
    #nfa_1's equivalences

    nfa_2_equivalences = [nfa_2.dup()]
    for __ in range(4):
        nfa_2_equivalences.append(shuffle_fa(nfa_2.dup()))
    for i in range(5):
        nfa_2.eliminateState(1)
        nfa_2_equivalences.append(nfa_2)
    #nfa_2's equivalences

    #Save GFAs
    for i in range(10):
        with open(f"./equivalence_test_data/gfa_{count}.pkl", "wb") as fp:
            pickle.dump(nfa_1_equivalences[i], fp)
        with open(f"./equivalence_test_data/gfa_{count + 10}.pkl", "wb") as fp:
            pickle.dump(nfa_2_equivalences[i], fp)
        count += 1

    for i in range(10):
        for j in range(i + 1, 10):
            writer.writerow([f"gfa_{count - 10 + i}.pkl", f"gfa_{count - 10 + j}.pkl", 1])
            writer.writerow([f"gfa_{count + i}.pkl", f"gfa_{count + j}.pkl", 1])
            writer.writerow([f"gfa_{count - 10 + i}.pkl", f"gfa_{count + j}.pkl", 0])
            writer.writerow([f"gfa_{count + i}.pkl", f"gfa_{count - 10 + j}.pkl", 0])
    count += 10

annotation_file.close()
