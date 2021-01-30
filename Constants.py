# Neural Network setup
INPUT_SIZE = 28
HIDDEN_LAYER_1 = 10
HIDDEN_LAYER_2 = 50
OUTPUT_SIZE = 3
LEARNING_RATE = 0.3
LOOK_BACK = 2
EPOCHS = 10

# Data processing
BINANCE_PAIR = 'BTCUSDT'
COIN_NAME = 'bitcoin'

# pre ~1100 vzoriek sa hodi FILTER_CONSTANT 100 (1100 vzoriek zodpoveda 5 minutovemu intervalu pre 96 hodin)
FILTER_CONSTANT = 100  # testovane aj s 25, 100 nam ale dava viac bodov (100 vyzera byt ideal pre nas pripad)

