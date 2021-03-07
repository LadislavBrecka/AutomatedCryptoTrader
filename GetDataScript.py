from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Constants import *
from Modules.services import MyLogger


DATASET_SAVE_FILE = 'Data/Datasets/data10.csv'
HOURS = 96
INTERVAL = '5m'


def download_dataset(hours, interval, file_name):
    binance = BinanceOhlcHandler(BINANCE_PAIR)
    binance.get_dataset(hours, interval)
    msg = "-----------------------------------------------------------------------------------------\n" + \
          "Actual price chart for {}. Close graph for continue!\n".format(BINANCE_PAIR) +\
          "-----------------------------------------------------------------------------------------"

    MyLogger.write(msg, file_name)
    binance.plot_candlestick(filter_const=FILTER_CONSTANT)

    MyLogger.write_console("Do you want to save this dataset for future use? [y]")
    if str(input()) == 'y':
        binance.save_to_csv(file_name)
        binance.print_to_file('Outputs/out1.txt')
        MyLogger.write_console("Dataset was saved to file {} !".format(file_name))
    else:
        MyLogger.write_console("Dataset was not saved!")
        pass


download_dataset(HOURS, INTERVAL, DATASET_SAVE_FILE)













#
#
# def load_dataset(file_name):
#     global binance
#     binance = BinanceOhlcHandler(BINANCE_PAIR)
#     binance.load_from_csv(file_name)
#     binance.print_to_file('Outputs/out1.txt')
#     print("----------------------------------------------------------------")
#     print("Dataset was loaded from file {}, close graph for continue!".format(file_name))
#     print("----------------------------------------------------------------")
#     binance.plot_candlestick(filter_const=FILTER_CONSTANT)







