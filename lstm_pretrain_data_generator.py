import csv
import random
from tqdm import tqdm
from itertools import combinations

from FAdo.reex import *
from FAdo.conversions import *

import alpha_zero.state_elimination.StateEliminationGame as game

from utils.random_nfa_generator import generate
from utils.heuristics import *

g = game.StateEliminationGame()

reduced = 1
minimization = False

def get_regex(gfa: GFA) -> list[RegExp]:
    random_order = [i for i in range(len(gfa.States)) if i != gfa.Initial and i not in gfa.Final]
    shuffle(random_order)
    bridge_state_name = decompose(gfa.dup())
    regex = set()
    #C1
    result = eliminate_randomly(gfa.dup(), minimization, random_order)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #C2
    result = eliminate_randomly(gfa.dup(), minimization, random_order, bridge_state_name)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #C3
    result = eliminate_by_state_weight_heuristic(gfa.dup(), minimization)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #C4
    result = eliminate_by_state_weight_heuristic(gfa.dup(), minimization, bridge_state_name)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #C5
    result = eliminate_by_repeated_state_weight_heuristic(gfa.dup(), minimization)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #C6
    result = eliminate_by_repeated_state_weight_heuristic(gfa.dup(), minimization, bridge_state_name)
    if reduced:
        result = result.reduced()
    regex.add(result)
    #Sanity check
    for combination in combinations(regex, 2):
        regex1, regex2 = combination
        assert str(regex1) != str(regex2)
    return list(regex)

def get_positive_cases(regex: list[GFA]) -> list[tuple]:
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


total_data_size = 1_350_000
progress_bar = tqdm(total=total_data_size)

with open("./annotations_file.csv", "w", newline='') as fp:
    writer = csv.writer(fp)
    data_size = 0
    k = 5
    d = 0.1
    while data_size < 1_350_000:
        reduced = random.randint(0, 1)
        n = random.randint(3, 10)
        first_gfa = generate(n, k, d, "nfa")
        second_gfa = generate(n, k, d, "nfa")
        if first_gfa == second_gfa:
            continue
        first_gfa = convert_nfa_to_gfa(first_gfa)
        second_gfa = convert_nfa_to_gfa(second_gfa)
        first_regex = get_regex(first_gfa)
        second_regex = get_regex(second_gfa)
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
        data_size += equal_length * 2
        progress_bar.update(equal_length * 2)
        CToken.clear_memory()
