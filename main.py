import services
import Teacher
import numpy as np
import NeuralNetwork_2_hidden as nn_2_hidden
from Constants import *
import pandas as pd
import DataHandling

'''
Defining neural network and services
'''
nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
splitter = services.Splitter()
normalizer = services.Normalize(DataHandling.binance.dataset)

'''
Finding and setting buy/sell signals, generating target values
'''
x = Teacher.generate_buy_sell_signal(DataHandling.binance.dataset, FILTER_CONSTANT)
targets = Teacher.get_target(len(DataHandling.binance.dataset), x)

# # For revision
# binance.dataset['Buy'] = x[0]
# binance.dataset['Sell'] = x[1]
# binance.dataset['Target'] = targets

'''
Normalizing whole dataset once
'''
temp_open = []
temp_high = []
temp_low = []
temp_close = []
temp_ema_diff = []
temp_momentum = []
temp_rsi = []
for i in range(len(DataHandling.binance.dataset)):
    row = DataHandling.binance.dataset.iloc[i]
    del row['Date']
    # del row['Buy']
    # del row['Sell']
    # del row['Target']
    row = row.values
    norm_row = normalizer.get_normalized_row(row)
    temp_open.append(norm_row[0])           # open
    temp_high.append(norm_row[1])           # high
    temp_low.append(norm_row[2])            # low
    temp_close.append(norm_row[3])          # close
    temp_ema_diff.append(norm_row[7])       # ema_short - ema_long
    temp_momentum.append(norm_row[10])      # momentum
    temp_rsi.append(norm_row[11])           # rsi

norm_dataset = pd.DataFrame(index=DataHandling.binance.dataset['Date'],
                            columns=['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_EmaDiff', 'Norm_Momentum', 'Norm_Rsi'])
norm_dataset['Norm_Open'] = temp_open
norm_dataset['Norm_High'] = temp_high
norm_dataset['Norm_Low'] = temp_low
norm_dataset['Norm_Close'] = temp_close
norm_dataset['Norm_EmaDiff'] = temp_ema_diff
norm_dataset['Norm_Momentum'] = temp_momentum
norm_dataset['Norm_Rsi'] = temp_rsi

'''
Splitting dataset to train and test datasets
'''
train, test = splitter.split_test_train(norm_dataset, SPLIT_TRAIN_TEST)

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

        nn.train(inputs, targets[i])
        print(i, e)

'''
Querying test dataset for testing Feed Forward Neural Network
'''
profit = 0.0
output_file = open('output.txt', 'w')
# Securing buy-sell-buy-sell pattern, no buy-buy or sell-sell is allowed
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
        print(ans)

        if ans[0] > 0.3:
            if flag != 0:
                profit = profit - DataHandling.binance.dataset['Close'][len(train)+i]
                print("Buying")
                output_file.write("{}\n".format("Buying"))
                output_file.write("{}\n".format(DataHandling.binance.dataset.iloc[len(train)+i]['Date']))
                output_file.write("{}\n".format(ans))
                flag = 0
            else:
                print("Holding")
                output_file.write("{}\n".format("Holding"))
                output_file.write("{}\n".format(ans))
        elif ans[1] > 0.3:
            if flag != 1:
                profit = profit + DataHandling.binance.dataset['Close'][len(train)+i]
                print("Selling")
                output_file.write("{}\n".format("Selling"))
                output_file.write("{}\n".format(DataHandling.binance.dataset.iloc[len(train) + i]['Date']))
                output_file.write("{}\n".format(ans))
                flag = 1
            else:
                print("Holding")
                output_file.write("{}\n".format("Holding"))
                output_file.write("{}\n".format(ans))
        else:
            print("Holding")
            output_file.write("{}\n".format("Holding"))
            output_file.write("{}\n".format(ans))

        print("Profit is {}".format(profit))
        output_file.write("Profit is {}\n".format(profit))

output_file.close()

'''
Plotting and printing
'''
DataHandling.binance.plot_candlestick(indicators=True, buy_sell=x, norm=norm_dataset, filter_const=FILTER_CONSTANT)
DataHandling.binance.print_to_file('out1.txt')


