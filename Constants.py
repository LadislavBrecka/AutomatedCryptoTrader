# Neural Network setup
INPUT_SIZE = 28         # look back = 2
HIDDEN_LAYER_1 = 20     # 2/3 of inputs size
HIDDEN_LAYER_2 = 15     # 1/2 - 2/3 of preceding layer
OUTPUT_SIZE = 3
LEARNING_RATE = 0.3
LOOK_BACK = 2           # look back value
EPOCHS = 3

# Data processing
BINANCE_PAIR = 'BTCUSDT'
COIN_NAME = 'bitcoin'

# pre ~1100 vzoriek sa hodi FILTER_CONSTANT 100 (1100 vzoriek zodpoveda 5 minutovemu intervalu pre 96 hodin)
FILTER_CONSTANT = 200  # higher value, more points of min/max will be found

