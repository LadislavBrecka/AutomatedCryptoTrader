from BinanceOhlcHandler import BinanceOhlcHandler
from services import Services
from CoinMarketCapHandler import CoinMarketCapHandler
import Teacher
import numpy as np
import NeuralNetwork_2_hidden as nn_2_hidden
import NeuralNetwork_2_hidden as nn_1_hidden
from Constants import *
import time

'''
Setting and getting dataset
'''

binance = BinanceOhlcHandler(BINANCE_PAIR)
binance.get_dataset()
# coinmarket = CoinMarketCapHandler(COIN_NAME)
# coinmarket.get_actual()
# # print(coinmarket.actual)

nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
# nn = nn_1_hidden.NeuralNetwork(24, 50, 1, 0.3)

services = Services()

'''
Finding and setting buy/sell signals
'''
x = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
targets = Teacher.get_target(len(binance.dataset), x)

# For revision
binance.dataset['Buy'] = x[0]
binance.dataset['Sell'] = x[1]
binance.dataset['Target'] = targets
'''
Plotting and printing
'''
binance.plot_candlestick(indicators=True, buy_sell=x, filter_const=FILTER_CONSTANT)

binance.print_to_file('out1.txt')

'''
Pushing new samples and the end of dataset
'''
# print("Printing from main.py: Start adding new samples")
# i = 0
# while i != 3:
#     time.sleep(60)
#     binance.get_recent_OHLC()
#     i += 1
#
# binance.print_to_file('out1.txt')

'''
Feed Forward neural network
'''

# for e in range(epochs):
#     for i in range(len(binance.dataset)):
#         t = binance.dataset.iloc[i]
#         tm1 = binance.dataset.iloc[i-1]
#         del t['Date']
#         del tm1['Date']
#         t = t.values
#         tm1 = tm1.values
#
#         inputs = np.concatenate([t, tm1])
#         inputs = inputs.reshape(-1, 1)
#         inputs = services.normalize(inputs)
#         inputs = inputs.reshape(28, )
#         inputs = inputs * 0.99 + 0.01
#         #
#         # print("Input value -> index {}\n value {}".format(i, inputs))
#         # print("Target value -> index {}\n value {}".format(i, targets[i]))
#
#         nn.train(inputs, targets[i])
#         print(i, e)


# k = binance.dataset.iloc[45]
# km = binance.dataset.iloc[44]
# del k['Date']
# del km['Date']
# k = k.values
# km = km.values
# inp = np.concatenate([k, km])
# inp = inp.reshape(-1, 1)
# inp = services.normalize(inp)
# inp = inp.reshape(28, )
# inp = inp * 0.99 + 0.01
#
# ans = nn.query(inp)
#
# print("\n")
# print(ans)



'''
LSTM neural network
'''
