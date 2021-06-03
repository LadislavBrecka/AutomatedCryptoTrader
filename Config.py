"""
                                        CONFIG FILE FOR TRADING APPLICATION
"""

"""
Neural network
"""
INPUT_SIZE = 42                     # number of inputs * look back constant - must be set manual
HIDDEN_LAYER_1 = 20
HIDDEN_LAYER_2 = 20
OUTPUT_SIZE = 2                     # [1.0 0.01] - buy, [0.01 1.0] - sell, [0.01 0.01] - hold

LOOK_BACK = 3
LEARNING_RATE = 0.08
EPOCHS = 50

HOLD_ERROR_PENALIZING = 0.01
NN_OUT_ANS_BUY_THRESHOLD = 0.5
NN_OUT_ANS_SELL_THRESHOLD = 0.5
FEE_PERCENTAGE = 0.001              # 1% from buy price (BINANCE)
BUY_QUANTITY = 0.03                 # 3% from price of crypto, e.q. price of ETH is 1300 EUR, we will be buying for 39 EUR

"""
Dataset
"""
BINANCE_PAIR = 'ETHEUR'             # for Binance handler
COIN_NAME = 'ethereum'              # for CoinMarketCap handler, not used in project
INTERVAL = '5m'                     # no other interval is supported, do not change!

"""
Teacher
"""
FILTER_CONSTANT = 30                # higher value, more points of min/max will be found

LOOK_AHEAD_FILTERED_SERIES = 10     # look ahead for better peak in 1st step on filtered series
                                    # Keep smaller on filtered series than on raw series, rather use LOOK_AHEAD_RAW_SERIES for tuning peaks

DELTA_FILTERED_SERIES = 2           # x delta, e.g. 5 mean only every 5 on x axes can be peak

LOOK_AHEAD_RAW_SERIES = 10          # look ahead for better peak in 2nd step on raw series

DELTA_RAW_SERIES = 15.0             # y delta, e.g. DELTA_RAW_SERIES = 25.0 mean 25.0 dollars is minimum between buy and sell

RSI_LOW = 40.0                      # rsi at which crypto turn to oversold
RSI_HIGH = 60.0                     # rsi at which crypto turn to overbought

"""
Normalizing parameters
"""
PRICE_AMPLITUDE = 1.2
VOLUME_AMPLITUDE = 1.0
DIFF_OPEN_CLOSE_AMPLITUDE = 1.0
EMA_AMPLITUDE = 1.2
DIF_EMA_AMPLITUDE = 0.7
MACD_AMPLITUDE = 1.2
MOMENTUM_AMPLITUDE = 0.7
GRADIENT_AMPLITUDE = 0.7

"""
Binance account API information
"""
API_KEY = 'wzDZXGFp9DK9QOChVTEOhZKaYEVAbbCBhsBDJXvSI78t2WlD3TqQhViWb8LH5Xom'            # PUT HERE YOUR API KEY FROM YOUR BINANCE ACCOUNT
API_SECRET = 'U6Gk59emVhyiNtuX9GDkpgtRwZlsAJkVhUFFzb3YvJEnN8NRDECozDnE8SgyT0kh'         # PUT HERE YOUR API SECRET FROM YOUR BINANCE ACCOUNT


