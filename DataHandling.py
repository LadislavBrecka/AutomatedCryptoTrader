from BinanceOhlcHandler import BinanceOhlcHandler
from Constants import *

binance = None
file = 'data1.csv'


def save_dataset():
    global binance
    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.get_dataset()
    print("-----------------------------------------------------------------------------------------")
    print("Actual price chart for {}. Close graph for continue!".format(BINANCE_PAIR))
    print("-----------------------------------------------------------------------------------------")
    binance.plot_candlestick(filter_const=FILTER_CONSTANT)
    print("Do you want to save this dataset for future use? [y]")
    if str(input()) == 'y':
        binance.save_to_csv(file)
        binance.print_to_file('out1.txt')
        print("Dataset was saved to file {} !".format(file))
    else:
        print("Dataset was not saved!")
        pass


def load_dataset():
    global binance
    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.load_from_csv(file)
    binance.print_to_file('out1.txt')
    print("----------------------------------------------------------------")
    print("Dataset was loaded from file {}, close graph for continue!".format(file))
    print("----------------------------------------------------------------")
    binance.plot_candlestick(filter_const=FILTER_CONSTANT)


load_dataset()
