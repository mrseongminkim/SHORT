import csv
import pickle

from FAdo.fa import NFA

from utils.random_nfa_generator import generate
from utils.fadomata import convert_nfa_to_gfa

annotation_file = open("./membership_test_data/annotations_file.csv", "w", newline="")
writer = csv.writer(annotation_file)

#there will be 2 * number_of_words_for_each_fa for a NFA (accept / reject)
number_of_words_for_each_fa = 5_000
count = 0
#there will be (2 * number_of_pairs) NFAs
number_of_pairs = 5

for _ in range(number_of_pairs):
    nfa_1: NFA = generate(7, 5, 0.1, "nfa")
    nfa_1.Sigma = {'0', '1', '2', '3', '4', '5'}
    nfa_1_accept_string = nfa_1.enumNFA(10)
    nfa_1_reject_string = []

    nfa_2: NFA = generate(7, 5, 0.1, "nfa")
    nfa_2.Sigma = {'0', '1', '2', '3', '4', '5'}
    nfa_2_accept_string = nfa_2.enumNFA(10)
    nfa_2_reject_string = []

    for nfa_2_string in nfa_2_accept_string:
        if not nfa_1.evalWordP(nfa_2_string):
            nfa_1_reject_string.append(nfa_2_string)

    for nfa_1_string in nfa_1_accept_string:
        if not nfa_2.evalWordP(nfa_1_string):
            nfa_2_reject_string.append(nfa_1_string)

    nfa_1_accept_string = nfa_1_accept_string[-number_of_words_for_each_fa:]
    nfa_1_reject_string = nfa_1_reject_string[-number_of_words_for_each_fa:]
    nfa_2_accept_string = nfa_2_accept_string[-number_of_words_for_each_fa:]
    nfa_2_reject_string = nfa_2_reject_string[-number_of_words_for_each_fa:]

    assert len(nfa_1_accept_string) == number_of_words_for_each_fa
    assert len(nfa_2_accept_string) == number_of_words_for_each_fa
    assert len(nfa_1_reject_string) == number_of_words_for_each_fa
    assert len(nfa_2_reject_string) == number_of_words_for_each_fa

    gfa_1 = convert_nfa_to_gfa(nfa_1)
    with open(f"./membership_test_data/gfa_{count}.pkl", "wb") as fp:
        pickle.dump(gfa_1, fp)
    count += 1
    gfa_2 = convert_nfa_to_gfa(nfa_2)
    with open(f"./membership_test_data/gfa_{count}.pkl", "wb") as fp:
        pickle.dump(gfa_2, fp)
    count += 1

    for i in range(number_of_words_for_each_fa):
        writer.writerow([f"gfa_{count - 2}.pkl", nfa_1_accept_string[i], 1])
        writer.writerow([f"gfa_{count - 2}.pkl", nfa_1_reject_string[i], 0])
        writer.writerow([f"gfa_{count - 1}.pkl", nfa_2_accept_string[i], 1])
        writer.writerow([f"gfa_{count - 1}.pkl", nfa_2_reject_string[i], 0])

annotation_file.close()
