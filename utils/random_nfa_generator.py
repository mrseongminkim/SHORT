import random
from pickle import dump
from typing import Union

import gmpy2
from FAdo.conversions import *

from utils.fadomata import *

from config import *

def make_fado_recognizable_nfa(n: int, k: int, nfa: gmpy2.mpz, finals: gmpy2.mpz, return_type: str="gfa") -> Union[GFA, NFA]:
    '''
    return a NFA/GFA object following below conditions
    1. 0 as initial and -1 as final -> no longer holds.
    2. reduced by lr equivalence relation
    3. order of states is randomly shuffled
    4. non-returning and non-exsiting
    5. initially connected
    6. single inital and final state
    7. name of states are always string
    8. trimmed
    '''
    fa = NFA()
    for i in range(n + 2):
        fa.addState()
    fa.addTransition(0, '@epsilon', 1)
    size = n * n * k
    for i in range(n):
        if finals.bit_test(i):
            fa.addTransition(i + 1, '@epsilon', n + 1)
    for i in range(size):
        if nfa.bit_test(i):
            src = i // (n * k)
            foo = i % (n * k)
            dst = foo // k
            lbl = foo % k
            fa.addTransition(src + 1, str(lbl), dst + 1)
    fa.setInitial({0})
    fa.setFinal({n + 1})

    initial_states = len(fa.States)
    original_fa = fa.dup()
    fa = fa.trim()
    trimmed_states = len(fa.States)

    if trimmed_states < initial_states:
        copied_fa = fa.dup()

    fa = fa.lrEquivNFA()
    reduced_fa = fa.dup()
    reduced_states = len(fa.States)

    '''NFA is reduced'''
    shuffle_fa(fa)
    shuffled_fa = fa.dup()

    rename_states(fa)
    renamed_fa = fa.dup()

    '''NFA is randomly sorted'''
    #gfa로 바꾸면 final state에 대해서 empty transition이 생긴다.
    if return_type == "gfa":
        fa = convert_nfa_to_gfa(fa)
        '''
        if trimmed_states < initial_states and reduced_states < trimmed_states:
            print("original:", original_fa.delta, original_fa.States)
            print("trimmed:", copied_fa.delta, copied_fa.States)
            print("reduced:", reduced_fa.delta, reduced_fa.States)
            print("shuffled:", shuffled_fa.delta, shuffled_fa.States)
            print("renamed:", renamed_fa.delta, renamed_fa.States)
            print("gfa:", fa.delta, fa.States, fa.Final)
            exit()
        '''
    return fa


def show_bitmap(nfa: gmpy2.mpz, size: int):
    for i in range(size):
        if nfa.bit_test(i):
            print('1', end='')
        else:
            print('0', end='')
    print()


def connect(n: int, k: int, nfa: gmpy2.mpz) -> gmpy2.mpz:
    b = n - 1
    visited = [0] * n
    unvisited = [i + 1 for i in range(b)]
    for a in range(1, n):
        foo = random.randint(0, a - 1)
        bar = random.randint(0, b - 1)
        src = visited[foo]
        dst = unvisited[bar]
        lbl = random.randint(0, k - 1)
        nfa = nfa.bit_set(src * n * k + dst * k + lbl)
        visited[a] = dst
        a += 1
        b -= 1
        del unvisited[bar]
    return nfa


def add_random_transitions(nfa: gmpy2.mpz, size: int, t: int) -> gmpy2.mpz:
    if not t:
        return nfa
    unused = []
    for i in range(size):
        if not nfa.bit_test(i):
            unused.append(i)
    j = len(unused)
    for i in range(t):
        foo = random.randint(0, j - 1)
        nfa = nfa.bit_set(unused[foo])
        del unused[foo]
        j -= 1
    return nfa


def generate(n: int, k: int, d: float, type: str="gfa"):
    size = n * n * k
    nfa = gmpy2.mpz()
    finals = gmpy2.mpz()
    if d < 0:
        d = random.random()
    nfa = connect(n, k, nfa)
    t = int(d * n * n * k - (n - 1))
    if t < 0:
        print(f"error: it is not possible to have an accessible NFA with {n} states and a transition density as low as {d}")
        exit()
    else:
        nfa = add_random_transitions(nfa, size, t)
    rstate = gmpy2.random_state(random.randint(0, 2147483647 - 1))
    finals = gmpy2.mpz_rrandomb(rstate, n)
    return make_fado_recognizable_nfa(n, k, nfa, finals, type)


def generate_test_nfas():
    for n in STATE:
        for k in ALPHABET:
            for d in DENSITY:
                file_name = "data/random_nfa/n%dk%dd%.1f.pkl" % (n, k, d)
                print(file_name)
                nfas = []
                for i in range(SAMPLE_SIZE):
                    nfa = generate(n, k, d)
                    nfas.append(nfa)
                with open(file_name, "wb") as fp:
                    dump(nfas, fp)
