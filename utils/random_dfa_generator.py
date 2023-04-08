import time

from FAdo.rndfap import *

seed = hash(time.perf_counter())

n = 7
k = 2
generator = ICDFArgen(n, k, seed=seed)

count = 0
while (count < 100):
    dfa = generator.next()
    if dfa.Final:
        count += 1
