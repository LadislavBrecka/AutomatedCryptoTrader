# Neural Network setup
INPUT_SIZE = 12         # look back = 3
HIDDEN_LAYER_1 = 20   # 3/4 of inputs size
HIDDEN_LAYER_2 = 20     # 2/3 - 3/4 of preceding layer
OUTPUT_SIZE = 2
LEARNING_RATE = 0.8
LOOK_BACK = 2           # look back value
EPOCHS = 500

# Data processing
BINANCE_PAIR = 'ETHEUR'
COIN_NAME = 'bitcoin'

# Min/Max peak finding
FILTER_CONSTANT = 150              # higher value, more points of min/max will be found
                                    # pre ~1100 vzoriek sa hodi FILTER_CONSTANT 100 (1100 vzoriek zodpoveda 5 minutovemu intervalu pre 96 hodin)

'''Keep smaller on filtered series than on raw series, rather use filter constant for tuning peaks on filtered series'''
LOOK_AHEAD_FILTERED_SERIES = 5     # look ahead for better peak in 1st step on filtered series
DELTA_FILTERED_SERIES = 2           # x delta, e.g. 5 mean only every 5 on x axes can be peak
LOOK_AHEAD_RAW_SERIES = 15         # look ahead for better peak in 2nd step on raw series
DELTA_RAW_SERIES = 1.0          # y delta, e.g. DELTA_RAW_SERIES = 25.0 mean 25.0 dollars is minimum between buy and sell
RSI_LOW = 40.0                        # rsi at which comodity turn to oversold
RSI_HIGH = 60.2                       # rsi at which comodity turn to overbought
