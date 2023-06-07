#Test parameters
SAMPLE_SIZE = 1_000

#NFA parameters
MIN_N = 3
MAX_N = 10
N_RANGE = MAX_N - MIN_N + 1
STATE = [n for n in range(MIN_N, MAX_N + 1)]
ALPHABET = [5]
DENSITY = [0.2]
