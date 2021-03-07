import Modules.services as services
import Modules.Teacher as Teacher
import numpy as np
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Constants import *
import pandas as pd
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Modules.services import MyLogger


DATASET_LOAD_FILE = 'Data/Datasets/3.3.csv'
LOAD_NN = True
SAVE_NN = False
SPLIT_TRAIN_TEST = 1.0


def nn_train():
    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    if LOAD_NN:
        nn.load_from_file()

    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.load_from_csv(DATASET_LOAD_FILE)

    splitter = services.Splitter()
    normalizer = services.Normalize(binance.dataset)

    '''
    Finding and setting buy/sell signals, generating target values
    '''
    x = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
    targets = Teacher.get_target(len(binance.dataset), x)

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
    for i in range(len(binance.dataset)):
        row = binance.dataset.iloc[i]
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

    norm_dataset = pd.DataFrame(index=binance.dataset['Date'],
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
    output_file = open('Outputs/output_testing.txt', 'w')
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

            if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
                if flag != 0:
                    buying_price = BUY_QUANTITY * binance.dataset['Close'][len(train)+i]
                    fees = (BUY_QUANTITY * binance.dataset['Close'][len(train)+i]) * FEE_PERCENTAGE
                    profit = profit - buying_price - fees
                    MyLogger.write("1. Buying at time : {}, for price {}.".format(binance.dataset.iloc[len(train)+i]['Date'], buying_price), output_file)
                    flag = 0
                else:
                    MyLogger.write("1. Holding", output_file)
            elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
                if flag != 1:
                    selling_price = BUY_QUANTITY * binance.dataset['Close'][len(train)+i]
                    fees = (BUY_QUANTITY * binance.dataset['Close'][len(train)+i]) * FEE_PERCENTAGE
                    profit = profit + selling_price - fees
                    MyLogger.write("1. Selling at time : {}, for price {}.".format(binance.dataset.iloc[len(train) + i]['Date'], selling_price), output_file)
                    flag = 1
                else:
                    MyLogger.write("1. Holding", output_file)
            else:
                MyLogger.write("1. Holding", output_file)

            MyLogger.write("2. Profit is {}\n".format(profit), output_file)

    msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" +\
          "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, binance.pair) + \
          "-----------------------------------------------------------------------------------------------------------------------------------"

    MyLogger.write(msg, output_file)
    output_file.close()

    '''
    Plotting and printing
    '''
    binance.plot_candlestick(indicators=True, buy_sell=x, norm=norm_dataset, filter_const=FILTER_CONSTANT)
    binance.print_to_file('Outputs/output_dataset.txt')

    if SAVE_NN:
        nn.save_to_file()


nn_train()


