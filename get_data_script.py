from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Constants import *
from Modules.services import MyLogger
import Modules.Teacher as Teacher


HOURS = 296
INTERVAL = '5m'


def download_dataset():
    if HOURS < 4*24:
        raise ValueError("Minimal hours for neural network to train is 96, which is 4 days!")

    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.get_dataset(HOURS, INTERVAL)
    msg = "-----------------------------------------------------------------------------------------\n" + \
          "Actual price chart for {}. Close graph for continue!\n".format(BINANCE_PAIR) +\
          "-----------------------------------------------------------------------------------------"

    MyLogger.write_console(msg)
    binance.plot_candlestick(filter_const=FILTER_CONSTANT)

    MyLogger.write_console("Do you want to save this dataset for future use? [y]")
    if str(input()) == 'y':
        datasets_by_day = [group[1] for group in binance.dataset.groupby(binance.dataset.index.date)]
        i = 0
        for frame in datasets_by_day:
            frame.to_csv("Data/Datasets/{}.csv".format(i))
            i = i + 1
        MyLogger.write_console("Number of datasets saved: {} !".format(len(datasets_by_day)))
    else:
        MyLogger.write_console("Dataset was not saved!")
        pass


def load_dataset(file_name):
    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.load_from_csv(file_name)
    '''
        Finding and setting buy/sell signals, generating target values
        '''
    ideal_signals = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)
    binance.plot_candlestick(indicators=True, buy_sell=ideal_signals, filter_const=FILTER_CONSTANT)


load_dataset('Data/Datasets/4.csv')







