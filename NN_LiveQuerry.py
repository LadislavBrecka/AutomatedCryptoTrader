import time
import numpy as np
from Constants import *
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
import Modules.services as services
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Modules.services import MyLogger
from datetime import datetime, timedelta
import Modules.Teacher as Teacher


def predict_live(predict_time):

    output_file = open('Outputs/live_log.txt', 'w')

    live_dataset_handler_binance = BinanceOhlcHandler(BINANCE_PAIR)
    live_dataset_handler_binance.get_dataset(12, INTERVAL)
    main_normalizer = services.Normalize(live_dataset_handler_binance.dataset)

    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    nn.load_from_file()

    MyLogger.write("Now going live, lets see!", output_file)
    t = 0
    flag = 1
    profit = 0.0
    ans_buy_list = []
    ans_sell_list = []
    ans_list = ans_buy_list, ans_sell_list

    for i in range(len(live_dataset_handler_binance.dataset)):
        ans_list[0].append(np.nan)
        ans_list[1].append(np.nan)

    while t != predict_time:

        now = datetime.utcnow()
        rounded = now - timedelta(minutes=now.minute % 5 - 5, seconds=now.second, microseconds=now.microsecond)
        wait_t = (rounded - now).seconds + 3
        MyLogger.write("Next time of execution is {} UTC, waiting {} seconds".format(rounded, wait_t), output_file)
        time.sleep(wait_t)
        MyLogger.write("\n---Execution start at {} UTC---".format(datetime.utcnow()), output_file)

        live_dataset_handler_binance.get_recent_OHLC()
        t += 1

        look_back_list = []

        for step in range(LOOK_BACK):
            entry_row = live_dataset_handler_binance.dataset.iloc[-1 - step]
            del entry_row['Date']
            entry_row = entry_row.values
            norm_entry_row = main_normalizer.get_normalized_row(entry_row)
            look_back_list.append(np.array(norm_entry_row))

        inputs = np.concatenate(look_back_list)
        inputs = inputs * 0.99 + 0.01

        ans = nn.query(inputs)

        if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
            if flag != 0:
                actual_price = live_dataset_handler_binance.get_actual_price()
                buying_price = BUY_QUANTITY * actual_price
                fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                profit = profit - buying_price - fees

                MyLogger.write("1. Buying at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], buying_price), output_file)
                MyLogger.write("2. Profit is {}\n".format(profit), output_file)
                ans_list[0].append(actual_price)
                ans_list[1].append(np.nan)
                flag = 0
            else:
                ans_list[0].append(np.nan)
                ans_list[1].append(np.nan)
                pass
        elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
            if flag != 1:
                actual_price = live_dataset_handler_binance.get_actual_price()
                selling_price = BUY_QUANTITY * actual_price
                fees = (BUY_QUANTITY * actual_price) * FEE_PERCENTAGE
                profit = profit + selling_price - fees

                MyLogger.write("1. Selling at time : {}, for price {}.".format(live_dataset_handler_binance.dataset.iloc[-1]['Date'], selling_price), output_file)
                MyLogger.write("2. Profit is {}\n".format(profit), output_file)
                ans_list[0].append(np.nan)
                ans_list[1].append(actual_price)
                flag = 1
            else:
                ans_list[0].append(np.nan)
                ans_list[1].append(np.nan)
                pass
        else:
            ans_list[0].append(np.nan)
            ans_list[1].append(np.nan)
            pass

        MyLogger.write("---Execution end at {} UTC---\n".format(datetime.utcnow()), output_file)

    msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" + \
          "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, live_dataset_handler_binance.pair) + \
          "-----------------------------------------------------------------------------------------------------------------------------------"

    MyLogger.write(msg, output_file)

    ideal_signals = Teacher.generate_buy_sell_signal(live_dataset_handler_binance.dataset, FILTER_CONSTANT)
    live_dataset_handler_binance.plot_candlestick(indicators=True, buy_sell=ideal_signals, answers=ans_list, filter_const=FILTER_CONSTANT)
    live_dataset_handler_binance.print_to_file('Outputs/output_live_dataset.txt')

    output_file.close()


predict_live(288)




