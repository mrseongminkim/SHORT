import random
import gmpy2

def show_bitmap(nfa: gmpy2.mpz, size: int):
    for i in range(size):
        if nfa.bit_test(i):
            print('1', end='')
        else:
            print('0', end='')
    print()

def make_fado_recognizable_nfa(n: int, k: int, nfa: gmpy2.mpz, finals: gmpy2.mpz, file_name: str):
    transition_list = ['0 @epsilon 1\n']
    size = n * n * k
    with open(file_name, 'a') as fp:
        fp.write('@NFA ')
        for i in range(n):
            if finals.bit_test(i):
                transition_list.append(f'{i + 1} @epsilon {n + 1}\n')
        fp.write(f'{n + 1} * 0\n')
        for i in range(size):
            if nfa.bit_test(i):
                src = i // (n * k)
                foo = i % (n * k)
                dst = foo // k
                lbl = foo % k
                transition_list.append(f'{src + 1} {lbl} {dst + 1}\n')
        transition_list.sort()
        for transition in transition_list:
            fp.write(transition)

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

def generate(n: int, k: int, d: float, file_name: str):
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
    make_fado_recognizable_nfa(n, k, nfa, finals, file_name)

def main():
    for n in range(3, 11):
        for k in [2, 5, 10]:
            for d in [0.2, 0.5]:
                file_name = '../data/n' + str(n) + 'k' + str(k) + ('s' if d == 0.2 else 'd') + '.txt'
                for i in range(20000):
                    generate(n, k, d, file_name)

main()