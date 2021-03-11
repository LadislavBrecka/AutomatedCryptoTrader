import time
import numpy as np
from Constants import *
from Modules.BinanceOhlcHandler import BinanceOhlcHandler
import Modules.services as services
import Modules.NeuralNetwork_2_hidden as nn_2_hidden
from Modules.services import MyLogger


PREDICT_TIME = 30
PREDICT_INTERVAL = '5m'


def predict_live():

    output_file = open('Outputs/output_live.txt', 'w')

    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.get_dataset(12, PREDICT_INTERVAL)

    normalizer = services.Normalize(binance.dataset)
    nn = nn_2_hidden.NeuralNetwork(INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_SIZE, LEARNING_RATE)
    nn.load_from_file()

    MyLogger.write("Now going live, lets see!", output_file)
    i = 0
    flag = 1
    profit = 0.0
    while i != PREDICT_TIME:
        MyLogger.write("waiting {} seconds".format(binance.interval*60), output_file)
        time.sleep(binance.interval*60)
        binance.get_recent_OHLC()
        i += 1

        t = binance.dataset.iloc[-1]
        tm1 = binance.dataset.iloc[-2]

        actual_close_price = t['Close']
        actual_date = t['Date']

        MyLogger.write("New data : {}, {} EUR".format(actual_date, actual_close_price), output_file)

        del t['Date']
        del tm1['Date']

        t = t.values
        tm1 = tm1.values

        t = normalizer.get_normalized_row(t)
        tm1 = normalizer.get_normalized_row(tm1)

        # Vymazat iba take indexi, ktore nechcem ako vstupy na nn
        # Pozriet si v NN_Training.py, ktore indexy reprezentuju ktore indikatori
        del t[12]
        del t[9]
        del t[8]
        del t[6]
        del t[5]
        del t[4]

        del tm1[12]
        del tm1[9]
        del tm1[8]
        del tm1[6]
        del tm1[5]
        del tm1[4]

        t = np.array(t)
        tm1 = np.array(tm1)
        inputs = np.concatenate([t, tm1])

        inputs = inputs * 0.99 + 0.01

        ans = nn.query(inputs)

        if ans[0] > NN_OUT_ANS_BUY_THRESHOLD:
            if flag != 0:
                buying_price = BUY_QUANTITY * actual_close_price
                fees = (BUY_QUANTITY * actual_close_price) * FEE_PERCENTAGE
                profit = profit - buying_price - fees
                MyLogger.write("1. Buying at time : {}, for price {}.".format(actual_date, buying_price), output_file)
                flag = 0
            else:
                MyLogger.write("1. Holding", output_file)
        elif ans[1] > NN_OUT_ANS_SELL_THRESHOLD:
            if flag != 1:
                selling_price = BUY_QUANTITY * actual_close_price
                fees = (BUY_QUANTITY * actual_close_price) * FEE_PERCENTAGE
                profit = profit + selling_price - fees
                MyLogger.write("1. Selling at time : {}, for price {}.".format(actual_date, selling_price), output_file)
                flag = 1
            else:
                MyLogger.write("1. Holding", output_file)
        else:
            MyLogger.write("1. Holding", output_file)

        MyLogger.write("2. Profit is {}\n".format(profit), output_file)

    msg = "-----------------------------------------------------------------------------------------------------------------------------------\n" + \
          "Final earning is {} EUR. All fees are included, application was buying for price 10% of actual price of {}.\n".format(profit, binance.pair) + \
          "-----------------------------------------------------------------------------------------------------------------------------------"

    MyLogger.write(msg, output_file)
    output_file.close()


predict_live()





