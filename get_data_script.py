from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Config import *
from Modules.services import MyLogger
import Modules.Teacher as Teacher
import sys


# Handler function for downloading dataset
def download_dataset(hours):
    if hours < 4*24:
        raise ValueError("Minimal hours for neural network to train is 96, which is 4 days!")

    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.get_dataset(hours, INTERVAL)
    msg = "-----------------------------------------------------------------------------------------\n" + \
          "Actual price chart for {}. Close graph for continue!\n".format(BINANCE_PAIR) +\
          "-----------------------------------------------------------------------------------------"

    MyLogger.write_console(msg)
    binance.plot_candlestick(filter_const=FILTER_CONSTANT)

    MyLogger.write_console("Do you want to save this dataset for future use? [y]")
    if str(input()) == 'y':
        datasets_by_day = [group[1] for group in binance.dataset.groupby(binance.dataset.index.date)]
        datasets_by_day.pop(0)
        datasets_by_day.pop(-1)
        i = 1
        for frame in datasets_by_day:
            frame.to_csv("Data/Datasets/Download/{}.csv".format(i))
            i = i + 1
        MyLogger.write_console("Number of datasets saved: {} !".format(len(datasets_by_day)))
    else:
        MyLogger.write_console("Dataset was not saved!")
        pass


# Handler function for loading dataset
def load_dataset(file_name):
    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.load_from_csv(file_name)

    # Finding and setting buy/sell signals, generating target values
    ideal_signals = Teacher.generate_buy_sell_signal(binance.dataset, FILTER_CONSTANT)

    binance.plot_candlestick(indicators=True, buy_sell=ideal_signals, filter_const=FILTER_CONSTANT)


'''
Console application, we must specify arguments in calling from terminal
'''
if len(sys.argv) == 1:
    raise ValueError("You must specify if you want to download (-d) or load (-l) file")
elif sys.argv[1] == '-d':
    try:
        download_dataset(int(sys.argv[2]))
    except Exception:
        raise ValueError("You did no specify hours")
elif sys.argv[1] == '-l':
    try:
        load_dataset(sys.argv[2])
    except Exception:
        raise ValueError("You did no specify existing relative path to file")
else:
    raise ValueError("You must specify if you want to download (-d) or load (-l) file")










