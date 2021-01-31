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
Finding and setting buy/sell signals, generating target values
'''
x = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
targets = Teacher.get_target(len(binance.dataset), x)

# For revision
# binance.dataset['Buy'] = x[0]
# binance.dataset['Sell'] = x[1]
# binance.dataset['Target'] = targets

'''
Splitting dataset to train and test datasets
'''
train, test = services.split_test_train(binance.dataset, 0.3)

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
Feed Forward neural network training
'''

for e in range(EPOCHS):
    for i in range(1, len(train)):
        t = train.iloc[i]
        tm1 = train.iloc[i-1]
        del t['Date']
        del tm1['Date']
        t = t.values
        tm1 = tm1.values

        inputs = np.concatenate([t, tm1])
        inputs = inputs.reshape(-1, 1)
        inputs = services.normalize(inputs)
        inputs = inputs.reshape(28, )
        inputs = inputs * 0.99 + 0.01

        # print("Input value -> index {}\n value {}".format(i, inputs))
        # print("Target value -> index {}\n value {}".format(i, targets[i]))

        nn.train(inputs, targets[i])
        print(i, e)

'''
Querying test dataset for testing Feed Forward Neural Network
'''
for i in range(1, len(test)):
        t = test.iloc[i]
        tm1 = test.iloc[i-1]
        del t['Date']
        del tm1['Date']
        t = t.values
        tm1 = tm1.values

        inputs = np.concatenate([t, tm1])
        inputs = inputs.reshape(-1, 1)
        inputs = services.normalize(inputs)
        inputs = inputs.reshape(28, )
        inputs = inputs * 0.99 + 0.01

        ans = nn.query(inputs)

        label = np.argmax(ans)

        # print(ans)
        # print(label)

        profit = 0.0

        if label == 0:
            profit = profit - test['Close'][i]
        elif label == 1:
            profit = profit + test['Close'][i]
        elif label == 2:
            pass

        print("Profit is {}".format(profit))

'''
LSTM neural network
'''
