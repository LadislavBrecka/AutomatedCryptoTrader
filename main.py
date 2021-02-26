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

temp_open = []
temp_high = []
temp_low = []
temp_close = []
temp_volume = []
temp_ema_diff = []
temp_macd_diff = []
temp_rsi = []
temp_gradient = []
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
    temp_open.append(norm_row[0])   # open
    temp_high.append(norm_row[1])   # high
    temp_low.append(norm_row[2])   # low
    temp_close.append(norm_row[3])   # close
    temp_volume.append(norm_row[4])   # volume
    temp_ema_diff.append(norm_row[5])   # ema_short - ema_long
    temp_macd_diff.append(norm_row[6])   # macd - signal
    temp_rsi.append(norm_row[7])        # rsi
    temp_gradient.append(norm_row[8])   # gradient

norm_dataset = pd.DataFrame(index=binance.dataset['Date'],
                            columns=['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_Macd_Diff', 'Norm_Rsi'])
norm_dataset['Norm_Open'] = temp_open
norm_dataset['Norm_High'] = temp_high
norm_dataset['Norm_Low'] = temp_low
norm_dataset['Norm_Close'] = temp_close
norm_dataset['Norm_Macd_Diff'] = temp_macd_diff
norm_dataset['Norm_Rsi'] = temp_rsi


'''
Splitting dataset to train and test datasets
'''
train, test = splitter.split_test_train(norm_dataset, 0.3)


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
        t = t.values
        tm1 = tm1.values

        t = np.array(t)
        tm1 = np.array(tm1)
        inputs = np.concatenate([t, tm1])

        inputs = (inputs * 0.99) + 0.01

        # print("Input value -> index {}\n value {}".format(i, inputs))
        # print("Target value -> index {}\n value {}".format(i, targets[i]))

        nn.train(inputs, targets[i])
        print(i, e)

'''
Querying test dataset for testing Feed Forward Neural Network
'''
profit = 0.0
output_file = open('output.txt', 'w')
# Flag is there for securing buy-sell-buy-sell pattern, no buy-buy or sell-sell is allowed
flag = 1

for i in range(1, len(test)):

        t = test.iloc[i]
        tm1 = test.iloc[i-1]
        t = t.values
        tm1 = tm1.values

        t = np.array(t)
        tm1 = np.array(tm1)
        inputs = np.concatenate([t, tm1])

        inputs = inputs * 0.99 + 0.01

        ans = nn.query(inputs)

        label = np.argmax(ans)

        print(ans)

        if ans[0] > 0.5:
            if flag != 0:
                profit = profit - binance.dataset['Close'][len(train)+i]
                print("Buying")
                output_file.write("{}\n".format("Buying"))
                output_file.write("{}\n".format(binance.dataset.iloc[len(train)+i]['Date']))
                flag = 0
            else:
                print("Holding")
                output_file.write("{}\n".format("Holding"))
        elif ans[1] > 0.5:
            if flag != 1:
                profit = profit + binance.dataset['Close'][len(train)+i]
                print("Selling")
                output_file.write("{}\n".format("Selling"))
                output_file.write("{}\n".format(binance.dataset.iloc[len(train) + i]['Date']))
                flag = 1
            else:
                print("Holding")
                output_file.write("{}\n".format("Holding"))
        else:
            print("Holding")
            output_file.write("{}\n".format("Holding"))

        print("Profit is {}".format(profit))
        output_file.write("Profit is {}\n".format(profit))

output_file.close()

'''
LSTM neural network
'''


'''
Plotting and printing
'''
binance.plot_candlestick(indicators=True, buy_sell=x, norm=norm_dataset, filter_const=FILTER_CONSTANT)
#
binance.print_to_file('out1.txt')
