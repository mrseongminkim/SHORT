import random
import gmpy2

from FAdo.conversions import *

from .fadomata import *

def show_bitmap(nfa: gmpy2.mpz, size: int):
    for i in range(size):
        if nfa.bit_test(i):
            print('1', end='')
        else:
            print('0', end='')
    print()

#Ensures non-returning, non-exiting, initially connected and single final state
def make_fado_recognizable_nfa(n: int, k: int, nfa: gmpy2.mpz, finals: gmpy2.mpz) -> GFA:
    gfa = NFA()
    for i in range(n + 2):
        gfa.addState()
    gfa.addTransition(0, '@epsilon', 1)
    size = n * n * k
    for i in range(n):
        if finals.bit_test(i):
            gfa.addTransition(i + 1, '@epsilon', n + 1)
    for i in range(size):
        if nfa.bit_test(i):
            src = i // (n * k)
            foo = i % (n * k)
            dst = foo // k
            lbl = foo % k
            gfa.addTransition(src + 1, str(lbl), dst + 1)
    gfa.setInitial({0})
    gfa.setFinal({n + 1})
    gfa = convert_nfa_to_gfa(gfa)
    gfa.reorder({(gfa.States).index(x) : int(x) for x in gfa.States})
    return gfa

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

def generate(n: int, k: int, d: float) -> GFA:
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
    return make_fado_recognizable_nfa(n, k, nfa, finals)

def main():
    for n in range(3, 11):
        for k in [2, 5, 10]:
            for d in [0.2, 0.5]:
                file_name = '../data/raw/n' + str(n) + 'k' + str(k) + ('s' if d == 0.2 else 'd') + '.txt'
                for i in range(100):
                    generate(n, k, d, file_name)

if __name__ == '__main__':
    generate(5, 5, 0.2)