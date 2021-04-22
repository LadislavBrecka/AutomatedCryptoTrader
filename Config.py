"""
Neural network
"""
INPUT_SIZE = 42                   # number of inputs * look back constant - must be set manual
HIDDEN_LAYER_1 = 20                 # 3/4 of inputs size
HIDDEN_LAYER_2 = 20                 # 2/3 - 3/4 of preceding layer
OUTPUT_SIZE = 2                     # [1.0 0.01] - buy
                                    # [0.01 1.0] - sellS
                                    # [0.01 0.01] - hold
LOOK_BACK = 3
LEARNING_RATE = 0.02
EPOCHS = 125                    # 12.3 - 175 is looking good for look_back=5, lr=0.25

HOLD_ERROR_PENALIZING = 0.01
NN_OUT_ANS_BUY_THRESHOLD = 0.5
NN_OUT_ANS_SELL_THRESHOLD = 0.5
FEE_PERCENTAGE = 0.001              # 1% from buy price (BINANCE)
BUY_QUANTITY = 0.03                  # 10% from price of ETH, e.q. price of ETH is 1300 EUR, we will be buying for 130 EUR

"""
Dataset
"""
BINANCE_PAIR = 'ETHEUR'
COIN_NAME = 'bitcoin'

INTERVAL = '5m'

"""
Min/Max peak finding
"""
FILTER_CONSTANT = 30                # higher value, more points of min/max will be found
                                    # 20.12 - pre ~1100 vzoriek sa hodi FILTER_CONSTANT 100 (1100 vzoriek zodpoveda 5 minutovemu intervalu pre 96 hodin)
                                    # 28.2 -  pre 720 vzoriek sa hodi FILTER_CONSTANT 30 (12 hodin 1-minutove vzorky)

LOOK_AHEAD_FILTERED_SERIES = 10      # look ahead for better peak in 1st step on filtered series
                                    # Keep smaller on filtered series than on raw series, rather use LOOK_AHEAD_RAW_SERIES for tuning peaks
                                    # 28.2 - najlepsia volba je 5, pretoze cim vacsie cislo, tym viac ukroji zo zaciatku aj konca (cim vacsie cislo tym viac ignoruje na zaciatku/konci)

DELTA_FILTERED_SERIES = 2           # x delta, e.g. 5 mean only every 5 on x axes can be peak
                                    # 28.2 - 2 je idealna volba

LOOK_AHEAD_RAW_SERIES = 10          # look ahead for better peak in 2nd step on raw series
                                    # 28.2 - 10 alebo 15 je ideal, ked je viac, uz preskakuje aj mozne body kupy/predaja kvoli tomu ze vpredu je lepsi peak

DELTA_RAW_SERIES = 15.0              # y delta, e.g. DELTA_RAW_SERIES = 25.0 mean 25.0 dollars is minimum between buy and sell
                                    # 28.2 - na ETH by to malo byt 25.0, treba vsetko vyladit pre tuto hodnotu

RSI_LOW = 40.0                      # rsi at which comodity turn to oversold
RSI_HIGH = 60.0                     # rsi at which comodity turn to overbought

"""
Normalizing parameters
"""
PRICE_AMPLITUDE = 1.2
VOLUME_AMPLITUDE = 1.0
DIFF_OPEN_CLOSE_AMPLITUDE = 1.0
EMA_AMPLITUDE = 1.2
DIF_EMA_AMPLITUDE = 0.7             # 28.2 - od 0.4 nastava konvergencia
MACD_AMPLITUDE = 1.2
MOMENTUM_AMPLITUDE = 0.7            # 28.2 - od 0.4 nastava konvergencia
GRADIENT_AMPLITUDE = 0.7


