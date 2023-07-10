#Test parameters
SAMPLE_SIZE = 1_00

#NFA parameters
MIN_N = 3
MAX_N = 7
N_RANGE = MAX_N - MIN_N + 1
STATE = [n for n in range(MIN_N, MAX_N + 1)]
ALPHABET = [5]
DENSITY = [0.2]

#CToken parameters
THRESHOLD = 5

#Random seed parameters
SEED = 210
