import Modules.services as services
import Modules.Teacher as Teacher
import numpy as np
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Config import *
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Modules.services import MyLogger


def nn_train(main_dataset_file_name, load_nn, save_nn, test_split, validation_dataset_file_name=None):
    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    if load_nn:
        nn.load_from_file()

    main_dataset_handler_binance = BinanceOhlcHandler(BINANCE_PAIR)
    main_dataset_handler_binance.load_from_csv(main_dataset_file_name)
    main_normalizer = services.Normalize(main_dataset_handler_binance.dataset)
    main_norm_dataset = main_normalizer.get_normalized_dataset(main_dataset_handler_binance.dataset)

    if validation_dataset_file_name is not None:
        validation_dataset_handler_binance = BinanceOhlcHandler(BINANCE_PAIR)
        validation_dataset_handler_binance.load_from_csv(validation_dataset_file_name)
        validation_normalizer = services.Normalize(validation_dataset_handler_binance.dataset)
        validation_norm_dataset = validation_normalizer.get_normalized_dataset(validation_dataset_handler_binance.dataset)
        validation_log_file = open('Outputs/validation_log.txt', 'w')

    '''
    Finding and setting buy/sell signals, generating target values
    '''
    ideal_signals = Teacher.generate_buy_sell_signal(main_dataset_handler_binance.dataset, FILTER_CONSTANT)
    targets = Teacher.get_target(len(main_dataset_handler_binance.dataset), ideal_signals)

    '''
    Splitting dataset to train and test datasets
    '''
    splitter = services.Splitter()
    train_dataset, test_dataset = splitter.split_test_train(main_norm_dataset, test_split)

    '''
    Feed Forward neural network training
    '''
    MyLogger.write_console("Starting training from file {} now!".format(main_dataset_file_name))

    for e in MyLogger.progressBar(range(EPOCHS), prefix='Progress:', suffix='Complete', length=50):
        for i in range(LOOK_BACK-1, len(train_dataset)):

            look_back_list = []
            for step in range(LOOK_BACK):
                look_back_list.append(np.array(train_dataset.iloc[i-step].values))

            inputs = np.concatenate(look_back_list)
            inputs = (inputs * 0.99) + 0.01
            nn.train(inputs, targets[i])

        if validation_dataset_file_name is not None:
            if e % 10 == 0:
                val_profit = nn_test_validation(nn, validation_dataset_handler_binance, validation_norm_dataset, validation_log_file, validation=True)
                MyLogger.write_file("For epoch number {} profit is {}".format(e, val_profit), validation_log_file)

    MyLogger.write_console("Training from file {} done!".format(main_dataset_file_name))
    if validation_dataset_file_name is not None:
        validation_log_file.close()

    '''
    Querying test dataset for testing Feed Forward Neural Network
    '''
    if test_split > 0.0:

        testing_log_file = open('Outputs/testing_log.txt', 'w')
        profit, ans_list_main = nn_test_validation(nn, main_dataset_handler_binance, test_dataset, testing_log_file)
        msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" + \
              "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, main_dataset_handler_binance.pair) + \
              "-----------------------------------------------------------------------------------------------------------------------------------"
        MyLogger.write(msg, testing_log_file)
        testing_log_file.close()
        '''
        Plotting and printing
       '''
        main_dataset_handler_binance.plot_candlestick(indicators=True, buy_sell=ideal_signals, answers=ans_list_main, norm=main_norm_dataset, filter_const=FILTER_CONSTANT)
        main_dataset_handler_binance.print_to_file('Outputs/output_dataset.txt')

    if save_nn:
        nn.save_to_file()


pass


def nn_test_validation(nn, dataset_handler_binance, test_norm_dataset, test_log_file, validation=False):

    train_dataset_len = len(dataset_handler_binance.dataset) - len(test_norm_dataset)
    testing_log_file = test_log_file

    '''
    Creating tuple for predicted buy/sell signals for plotting
    '''
    if not validation:
        ans_buy_list = []
        ans_sell_list = []
        ans_list = ans_buy_list, ans_sell_list

        # length of tuple must have same length as whole dataset
        # if testing was on same dataset as training, at the start we must fill np.nan so many times, as is length of
        # training dataset .. also, we must compensate LOOK_BACK constant at the start by filling np.nan
        for i in range(train_dataset_len + LOOK_BACK - 1):
            ans_list[0].append(np.nan)
            ans_list[1].append(np.nan)

    # Securing buy-sell-buy-sell pattern, no buy-buy or sell-sell is allowed
    flag = 1

    profit = 0.0

    # main test loop
    for i in range(LOOK_BACK - 1, len(test_norm_dataset)):

        look_back_list = []
        for step in range(LOOK_BACK):
            look_back_list.append(np.array(test_norm_dataset.iloc[i - step].values))

        inputs = np.concatenate(look_back_list)
        inputs = inputs * 0.99 + 0.01
        ans = nn.query(inputs)

        if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
            if flag != 0:
                buying_price = BUY_QUANTITY * dataset_handler_binance.dataset['Close'][train_dataset_len + i]
                fees = (BUY_QUANTITY * dataset_handler_binance.dataset['Close'][train_dataset_len + i]) * FEE_PERCENTAGE
                profit = profit - buying_price - fees
                if not validation:
                    MyLogger.write("1. Buying at time : {}, for price {}.".format(dataset_handler_binance.dataset.iloc[train_dataset_len + i]['Date'],buying_price), testing_log_file)
                    MyLogger.write("2. Profit is {}\n".format(profit), testing_log_file)
                    ans_list[0].append(dataset_handler_binance.dataset['Close'][train_dataset_len + i])
                    ans_list[1].append(np.nan)
                flag = 0
            else:
                if not validation:
                    ans_list[0].append(np.nan)
                    ans_list[1].append(np.nan)
                pass
        elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
            if flag != 1:
                selling_price = BUY_QUANTITY * dataset_handler_binance.dataset['Close'][train_dataset_len + i]
                fees = (BUY_QUANTITY * dataset_handler_binance.dataset['Close'][train_dataset_len+ i]) * FEE_PERCENTAGE
                profit = profit + selling_price - fees
                if not validation:
                    MyLogger.write("1. Selling at time : {}, for price {}.".format(dataset_handler_binance.dataset.iloc[train_dataset_len + i]['Date'], selling_price), testing_log_file)
                    MyLogger.write("2. Profit is {}\n".format(profit), testing_log_file)
                    ans_list[0].append(np.nan)
                    ans_list[1].append(dataset_handler_binance.dataset['Close'][train_dataset_len + i])
                flag = 1
            else:
                if not validation:
                    ans_list[0].append(np.nan)
                    ans_list[1].append(np.nan)
                pass
        else:
            if not validation:
                ans_list[0].append(np.nan)
                ans_list[1].append(np.nan)
            pass

    if validation:
        return profit
    else:
        return profit, ans_list






