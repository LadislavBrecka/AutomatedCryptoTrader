from BinanceOhlcHandler import BinanceOhlcHandler
import services
from CoinMarketCapHandler import CoinMarketCapHandler
import Teacher
import numpy as np
import NeuralNetwork_2_hidden as nn_2_hidden
import NeuralNetwork_2_hidden as nn_1_hidden
from Constants import *
import time
import pandas as pd

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

splitter = services.Splitter()
normalizer = services.Normalize(binance.dataset)

'''
Finding and setting buy/sell signals, generating target values
'''
x = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
targets = Teacher.get_target(len(binance.dataset), x)

# # For revision
binance.dataset['Buy'] = x[0]
binance.dataset['Sell'] = x[1]
binance.dataset['Target'] = targets
binance.dataset['Norm'] = np.nan

temp1 = []
temp2 = []
temp3 = []
temp4 = []
temp5 = []
temp6 = []
temp7 = []
for i in range(len(binance.dataset)):
    row = binance.dataset.iloc[i]
    del row['Date']
    del row['Ema3']
    del row['Ema6']
    del row['Macd']
    del row['Signal']
    del row['Momentum']
    del row['Buy']
    del row['Sell']
    del row['Target']
    row = row.values
    norm_row = normalizer.get_normalized_row(row)
    temp1.append(norm_row[0])
    temp2.append(norm_row[1])
    temp3.append(norm_row[2])
    temp4.append(norm_row[3])
    temp5.append(norm_row[4])
    temp6.append(norm_row[5])
    temp7.append(norm_row[6])

index = binance.dataset['Date']
columns = ['Norm_O', 'Norm_H', 'Norm_L', 'Norm_C', 'Norm_V', 'Norm_E', 'Norm_M']
temp_norm_dat = pd.DataFrame(index=index, columns=columns)
temp_norm_dat['Norm_O'] = temp1
temp_norm_dat['Norm_H'] = temp2
temp_norm_dat['Norm_L'] = temp3
temp_norm_dat['Norm_C'] = temp4
temp_norm_dat['Norm_V'] = temp5
temp_norm_dat['Norm_E'] = temp6
temp_norm_dat['Norm_M'] = temp7


'''
Splitting dataset to train and test datasets
'''
train, test = splitter.split_test_train(binance.dataset, 0.3)

'''
Plotting and printing
'''
binance.plot_candlestick(indicators=True, buy_sell=x, norm=temp_norm_dat, filter_const=FILTER_CONSTANT)

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
        del t['Ema3']
        del t['Ema6']
        del t['Macd']
        del t['Signal']
        del t['Momentum']
        del t['Buy']
        del t['Sell']
        del t['Target']

        del tm1['Date']
        del tm1['Ema3']
        del tm1['Ema6']
        del tm1['Macd']
        del tm1['Signal']
        del tm1['Momentum']
        del tm1['Buy']
        del tm1['Sell']
        del tm1['Target']
        t = t.values
        tm1 = tm1.values

        t_normalized = np.array(normalizer.get_normalized_row(t))
        tm1_normalized = np.array(normalizer.get_normalized_row(tm1))

        inputs = np.concatenate([t_normalized, tm1_normalized])

        inputs = inputs * 0.99 + 0.01

        print("Input value -> index {}\n value {}".format(i, inputs))
        print("Target value -> index {}\n value {}".format(i, targets[i]))

        nn.train(inputs, targets[i])
        print(i, e)

'''
Querying test dataset for testing Feed Forward Neural Network
'''
profit = 0.0
output_file = open('output.txt', 'w')
for i in range(1, len(test)):
        t = test.iloc[i]
        tm1 = test.iloc[i-1]
        del t['Date']
        del t['Ema3']
        del t['Ema6']
        del t['Macd']
        del t['Signal']
        del t['Momentum']
        del t['Buy']
        del t['Sell']
        del t['Target']

        del tm1['Date']
        del tm1['Ema3']
        del tm1['Ema6']
        del tm1['Macd']
        del tm1['Signal']
        del tm1['Momentum']
        del tm1['Buy']
        del tm1['Sell']
        del tm1['Target']
        t = t.values
        tm1 = tm1.values

        t_normalized = np.array(normalizer.get_normalized_row(t))
        tm1_normalized = np.array(normalizer.get_normalized_row(tm1))

        inputs = np.concatenate([t_normalized, tm1_normalized])
        inputs = inputs * 0.99 + 0.01

        ans = nn.query(inputs)

        label = np.argmax(ans)

        print(ans)

        if ans[0] > 0.2:
            profit = profit - test['Close'][i]
            print("Buying")
            output_file.write("{}\n".format("Buying"))
        elif ans[1] > 0.1:
            profit = profit + test['Close'][i]
            print("Selling")
            output_file.write("{}\n".format("Selling"))
        else:
            print("Holding")
            output_file.write("{}\n".format("Holding"))

        print("Profit is {}".format(profit))
        output_file.write("Profit is {}\n".format(profit))

'''
LSTM neural network
'''
