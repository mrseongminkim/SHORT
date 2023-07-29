import csv
import random
from itertools import combinations

from FAdo.reex import *
from FAdo.conversions import *

import alpha_zero.state_elimination.StateEliminationGame as game

from utils.random_nfa_generator import generate
from utils.heuristics import *

g = game.StateEliminationGame()

minimization = False

def get_regex(gfa: GFA) -> list[RegExp]:
    random_order = [i for i in range(1, len(gfa.States) - 1)]
    shuffle(random_order)
    bridge_state_name = decompose(gfa.dup())
    regex = set()
    #C1
    regex.add(eliminate_randomly(gfa.dup(), minimization, random_order))
    #C2
    regex.add(eliminate_randomly(gfa.dup(), minimization, random_order, bridge_state_name))
    #C3
    regex.add(eliminate_by_state_weight_heuristic(gfa.dup(), minimization))
    #C4
    regex.add(eliminate_by_state_weight_heuristic(gfa.dup(), minimization, bridge_state_name))
    #C5
    regex.add(eliminate_by_repeated_state_weight_heuristic(gfa.dup(), minimization))
    #C6
    regex.add(eliminate_by_repeated_state_weight_heuristic(gfa.dup(), minimization, bridge_state_name))
    #Sanity check
    for combination in combinations(regex, 2):
        regex1, regex2 = combination
        assert regex1.compare(regex2)
    #Reduced?
    return list(regex)

def get_positive_cases(regex: list) -> list[tuple]:
    list = []
    for combination in combinations(regex, 2):
        regex1, regex2 = combination
        regex1 = g.get_encoded_regex(regex1)
        regex2 = g.get_encoded_regex(regex2)
        list.append((regex1, regex2, 1))
    return list

def get_negative_rows(first_regex: list, second_regex: list) -> list[tuple]:
    list = []
    for i in range(0, len(first_regex)):
        regex1 = first_regex[i]
        regex1 = g.get_encoded_regex(regex1)
        for j in range(0, len(second_regex)):
            regex2 = second_regex[j]
            regex2 = g.get_encoded_regex(regex2)
            list.append((regex1, regex2, 0))
    return list

with open("./annotations_file.csv", "w", newline='') as fp:
    data_size = 0
    writer = csv.writer(fp)
    k = 5
    d = 0.1
    while data_size < 420_000:
        n = random.randint(3, 10)
        first_gfa = generate(n, k, d, "gfa")
        second_gfa = generate(n, k, d, "gfa")
        first_regex = get_regex(first_gfa)
        second_regex = get_regex(second_gfa)
        if first_regex[-1].compare(second_regex[-1]):
            print("These are actually the same")
            continue
        positive_cases = []
        negative_cases = []
        positive_cases += get_positive_cases(first_regex)
        positive_cases += get_positive_cases(second_regex)
        negative_cases += get_negative_rows(first_regex, second_regex)
        random.shuffle(positive_cases)
        random.shuffle(negative_cases)
        equal_length = min(len(positive_cases), len(negative_cases))
        positive_cases = positive_cases[:equal_length]
        negative_cases = negative_cases[:equal_length]

        writer.writerows(positive_cases)
        writer.writerows(negative_cases)
        data_size += equal_length
        print("data_size:", data_size)
        CToken.clear_memory()
print("done")
