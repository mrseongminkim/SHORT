from torch.cuda import is_available

#Test parameters
SAMPLE_SIZE = 10_000

#NFA parameters
MIN_N = 3
MAX_N = 10
N_RANGE = MAX_N - MIN_N + 1
STATE = [n for n in range(MIN_N, MAX_N + 1)]
ALPHABET = [5]
DENSITY = [0.2]

#CToken parameters
THRESHOLD = 5

#Random seed parameters
SEED = 210

#Neural network and MCTS parameters
NUMBER_OF_ITERATIONS = 20
NUMBER_OF_EPISODES = 100
TEMPERATURE_THRESHOLD = 0
MAXIMUM_LENGTH_OF_QUEUE = 200_000
NUMBER_OF_MCTS_SIMULATIONS = 25
CPUCT = 3
CHECKPOINT = "./alpha_zero/models/"
LOAD_MODEL = False
LOAD_FOLDER_FILE = ("./alpha_zero/models/", "best.pth.tar")
NUMBER_OF_ITERATIONS_FOR_TRAIN_EXAMPLES_HISTORY = 20
EPS = 1e-8
LR = 0.001
DROUPOUT = 0.0
EPOCHS = 20
BATCH_SIZE = 16
CUDA = is_available()
NUMBER_OF_CHANNELS = 32

#regex-board parameters
MAX_LEN = 50
VOCAB_SIZE = 16
EMBEDDING_DIMENSION = 2
LSTM_DIMENSION = 32
