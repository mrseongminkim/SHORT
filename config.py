from torch.cuda import is_available
from math import ceil

#Test parameters
SAMPLE_SIZE = 100

#NFA parameters
MIN_N = 3
MAX_N = 10
N_RANGE = MAX_N - MIN_N + 1
STATE = [n for n in range(MIN_N, MAX_N + 1)]
ALPHABET = [5]
DENSITY = [0.1]

#CToken parameters
THRESHOLD = 5

#Random seed parameters
SEED = 210

#Neural network and MCTS parameters
NUMBER_OF_ITERATIONS = 9999999999
NUMBER_OF_EPISODES = 1_00
TEMPERATURE = 2
MAXIMUM_LENGTH_OF_QUEUE = 200_000
NUMBER_OF_MCTS_SIMULATIONS = 100
CPUCT = 1.3
OPTIMAL_BONUS = 2
CHECKPOINT = "./alpha_zero/models/"
LOAD_MODEL = True
LOAD_FOLDER_FILE = ("./alpha_zero/models/", "checkpoint_3.pth.tar")
NUMBER_OF_ITERATIONS_FOR_TRAIN_EXAMPLES_HISTORY = 999999999999
EPS = 1e-8
LR = 0.001
DROUPOUT = 0.0
EPOCHS = 20
BATCH_SIZE = 128
CUDA = is_available()
NUMBER_OF_CHANNELS = 64

#regex-board parameters
MAX_LEN = 50
VOCAB_SIZE = 16
REGEX_EMBEDDING_DIMENSION = ceil(VOCAB_SIZE ** (1 / 4)) #2
LSTM_DIMENSION = 32

#GNN parameters
MAX_STATES = 50
STATE_NUMBER_EMBEDDING_DIMENSION = ceil((MAX_STATES + 2) ** (1 / 4)) #3
NUMBER_OF_HEADS = 4

VERBOSE = False
