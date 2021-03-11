import Modules.services as services
import Modules.Teacher as Teacher
import numpy as np
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Constants import *
import pandas as pd
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Modules.services import MyLogger
import time


def nn_train(dataset_file_name, load_nn, save_nn, test_split):
    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    if load_nn:
        nn.load_from_file()

    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.load_from_csv(dataset_file_name)

    splitter = services.Splitter()
    normalizer = services.Normalize(binance.dataset)

    '''
    Finding and setting buy/sell signals, generating target values
    '''
    ideal_signals = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
    targets = Teacher.get_target(len(binance.dataset), ideal_signals)

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
    temp_volume = []
    temp_emaShort = []
    temp_emaLong = []
    temp_ema_diff = []
    temp_macd = []
    temp_signal = []
    temp_momentum = []
    temp_rsi = []
    temp_gradient = []
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
        temp_volume.append(norm_row[4])       # volume
        temp_emaShort.append(norm_row[5])     # ema short
        temp_emaLong.append(norm_row[6])      # ema long
        temp_ema_diff.append(norm_row[7])       # ema_short - ema_long
        temp_macd.append(norm_row[8])         # macd
        temp_signal.append(norm_row[9])       # signal
        temp_momentum.append(norm_row[10])      # momentum
        temp_rsi.append(norm_row[11])           # rsi
        temp_gradient.append(norm_row[12])    # gradient

    norm_dataset = pd.DataFrame(index=binance.dataset['Date'],
                                columns=['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_Volume','Norm_EmaShort', 'Norm_EmaLong', 'Norm_EmaDiff', 'Norm_Macd', 'Norm_Signal', 'Norm_Momentum', 'Norm_Rsi', 'Norm_Gradient'])

    norm_dataset['Norm_Open'] = temp_open
    norm_dataset['Norm_High'] = temp_high
    norm_dataset['Norm_Low'] = temp_low
    norm_dataset['Norm_Close'] = temp_close
    norm_dataset['Norm_Volume'] = temp_volume
    norm_dataset['Norm_EmaShort'] = temp_emaShort
    norm_dataset['Norm_EmaLong'] = temp_emaLong
    norm_dataset['Norm_EmaDiff'] = temp_ema_diff
    norm_dataset['Norm_Macd'] = temp_macd
    norm_dataset['Norm_Signal'] = temp_signal
    norm_dataset['Norm_Momentum'] = temp_momentum
    norm_dataset['Norm_Rsi'] = temp_rsi
    norm_dataset['Norm_Gradient'] = temp_gradient

    '''
    Splitting dataset to train and test datasets
    '''
    train_dataset, test_dataset = splitter.split_test_train(norm_dataset, test_split)

    '''
    Creating tuple for predicted buy/sell signals for plotting 
    '''
    ans_buy_list = []
    ans_sell_list = []
    ans_list = ans_buy_list, ans_sell_list

    # if testing is required on some dataset as training, fill tuple with np.nan as many times, as is length of training dataset
    # because length of tuple must have same length as if whole dataset was for testing
    if test_split > 0.0:
        for i in range(len(train_dataset) + LOOK_BACK - 1):
            ans_list[0].append(np.nan)
            ans_list[1].append(np.nan)

    '''
    Feed Forward neural network training
    '''
    MyLogger.write_console("Starting training from file {} now!".format(dataset_file_name))

    for e in MyLogger.progressBar(range(EPOCHS), prefix='Progress:', suffix='Complete', length=50):
        for i in range(LOOK_BACK-1, len(train_dataset)):

            look_back_list = []
            for step in range(LOOK_BACK):
                look_back_list.append(np.array(train_dataset.iloc[i-step].values))

            inputs = np.concatenate(look_back_list)
            inputs = (inputs * 0.99) + 0.01
            nn.train(inputs, targets[i])

    MyLogger.write_console("Training from file {} done!".format(dataset_file_name))

    '''
    Querying test dataset for testing Feed Forward Neural Network
    '''
    profit = 0.0
    testing_log_file = open('Outputs/testing_log.txt', 'w')
    # Securing buy-sell-buy-sell pattern, no buy-buy or sell-sell is allowed
    flag = 1

    for i in range(LOOK_BACK-1, len(test_dataset)):

        look_back_list = []
        for step in range(LOOK_BACK):
            look_back_list.append(np.array(test_dataset.iloc[i - step].values))

        inputs = np.concatenate(look_back_list)
        inputs = inputs * 0.99 + 0.01
        ans = nn.query(inputs)

        if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
            if flag != 0:
                buying_price = BUY_QUANTITY * binance.dataset['Close'][len(train_dataset)+i]
                fees = (BUY_QUANTITY * binance.dataset['Close'][len(train_dataset)+i]) * FEE_PERCENTAGE
                profit = profit - buying_price - fees
                MyLogger.write("1. Buying at time : {}, for price {}.".format(binance.dataset.iloc[len(train_dataset)+i]['Date'], buying_price), testing_log_file)
                MyLogger.write("2. Profit is {}\n".format(profit), testing_log_file)
                flag = 0
                ans_list[0].append(binance.dataset['Close'][len(train_dataset)+i])
                ans_list[1].append(np.nan)
            else:
                ans_list[0].append(np.nan)
                ans_list[1].append(np.nan)
                pass
        elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
            if flag != 1:
                selling_price = BUY_QUANTITY * binance.dataset['Close'][len(train_dataset)+i]
                fees = (BUY_QUANTITY * binance.dataset['Close'][len(train_dataset)+i]) * FEE_PERCENTAGE
                profit = profit + selling_price - fees
                MyLogger.write("1. Selling at time : {}, for price {}.".format(binance.dataset.iloc[len(train_dataset) + i]['Date'], selling_price), testing_log_file)
                MyLogger.write("2. Profit is {}\n".format(profit), testing_log_file)
                flag = 1
                ans_list[0].append(np.nan)
                ans_list[1].append(binance.dataset['Close'][len(train_dataset) + i])
            else:
                ans_list[0].append(np.nan)
                ans_list[1].append(np.nan)
                pass
        else:
            ans_list[0].append(np.nan)
            ans_list[1].append(np.nan)
            pass

    if test_split > 0.0:
        msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" +\
              "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, binance.pair) + \
              "-----------------------------------------------------------------------------------------------------------------------------------"

        MyLogger.write(msg, testing_log_file)
        testing_log_file.close()

        '''
        Plotting and printing
        '''
        binance.plot_candlestick(indicators=True, buy_sell=ideal_signals, answers=ans_list, norm=norm_dataset, filter_const=FILTER_CONSTANT)

        binance.print_to_file('Outputs/output_dataset.txt')

    if save_nn:
        nn.save_to_file()


pass






