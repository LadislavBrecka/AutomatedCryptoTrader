import pandas as pd
import numpy as np
from services import Services
import peakdetect


# Generate target list of format:
# [1 0.01 0.01] - buy
# [0.01 1 0.01] - sell
# [0.01 0.01 1] - hold


def get_target(dataset_length: int, buy_sell: tuple) -> list:
    targets = []

    for i in range(dataset_length):

        desire_output = []
        if not pd.isnull(buy_sell[0][i]):
            desire_output.append(1)
            desire_output.append(0.01)
            desire_output.append(0.01)
        elif not pd.isnull(buy_sell[1][i]):
            desire_output.append(0.01)
            desire_output.append(1)
            desire_output.append(0.01)
        else:
            desire_output.append(0.01)
            desire_output.append(0.01)
            desire_output.append(1)

        targets.append(desire_output)

    return targets


# Most important function of our application, written entirely by ME!
def generate_buy_sell_signal(dataset: pd.DataFrame, filter_const: int) -> tuple:

    # Low pass filter on closed price, for finding cleaner min and max
    # Greater value of filter constant give us more buy/sell signals,
    # because of less filtered values
    filtered_price = Services.fft_filter(dataset['Close'], filter_const)

    # Detecting min and max peaks in filtered series
    peaks = peakdetect.peakdetect(np.array(filtered_price), lookahead=10, delta=10)

    # peakdetect function return tuple of min and max lists
    # [[[ind, val], [ind, val], [ind, val]],
    #  [[ind, val], [ind, val], [ind, val]]
    # ]
    # Min/max lists are in format list of lists (each min/max value in list has [ind, val])
    # Parsing peaks to retrieve only indexes of min/max values from min/max lists and push them to corresponded lists
    max_peaks_ind = []
    for l in peaks[0]:
        # Appending index of found maximum to list of indexes of max values
        max_peaks_ind.append(l[0])
    min_peaks_ind = []
    for l in peaks[1]:
        # Appending index of found minimum to list of indexes of min values
        min_peaks_ind.append(l[0])

    # We get indexes of min/max values in filtered series, which have a bit offset from min/max in non-filtered series
    # Now we need to get precised min/max in non-filtered series by scanning in non-filtered series few samples
    # to each side and scan in this interval for new min/max indexes
    PRECISION_CONSTANT = 10

    # For every index in max indexes founded in filtered series, look at non-filtered series,
    # and in range of PRECISION CONSTANT to each side scan for better max
    # Output from np.where was a bit mess, sometimes it returns tuple, sometimes empty array
    # sometimes 2 values,so there must be proper checking for proper case
    new_max_ind = []
    for i in max_peaks_ind:
        k = np.where(dataset['Close'].values == dataset['Close'][i-PRECISION_CONSTANT:i+PRECISION_CONSTANT].max())
        if isinstance(k, tuple):
            if k[0].size == 0:
                new_max_ind.append(i)
            else:
                new_max_ind.append(k[0])
        else:
            if k.size == 0:
                new_max_ind.append(i)
            else:
                new_max_ind.append(k)

    # For every index in mi indexes founded in filtered series, look at non-filtered series,
    # and in range of PRECISION CONSTANT to each side scan for better min
    # Output from np.where was a bit mess, sometimes it returns tuple, sometimes empty array
    # sometimes 2 values,so there must be proper checking for proper case
    new_min_ind = []
    for i in min_peaks_ind:
        k = np.where(dataset['Close'].values == dataset['Close'][i - PRECISION_CONSTANT:i + PRECISION_CONSTANT].min())
        if isinstance(k, tuple):
            if k[0].size == 0:
                new_min_ind.append(i)
            else:
                new_min_ind.append(k[0])
        else:
            if k.size == 0:
                new_min_ind.append(i)
            else:
                new_min_ind.append(k)

    sig_price_buy = []
    sig_price_sell = []
    # Flag is there for securing buy-sell-buy-sell pattern, no buy-buy or sell-sell is allowed
    flag = 1
    budget = 0

    for i in range(0, len(dataset)):
        # Detecting sell signals, if there is maximum on non-filtered series and RSI > 40
        if i in new_max_ind and dataset['Rsi'][i] > 40:
            if flag != 1:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(dataset['Close'][i])
                flag = 1
                budget = budget + dataset['Close'][i]
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)

        # Detecting buy signals, if there is minimum on non-filtered series and RSI > 40
        elif i in new_min_ind and dataset['Rsi'][i] < 60:
            if flag != 0:
                sig_price_sell.append(np.nan)
                sig_price_buy.append(dataset['Close'][i])
                flag = 0
                budget = budget - dataset['Close'][i]
            else:
                sig_price_buy.append(np.nan)
                sig_price_sell.append(np.nan)

        # Handling nan values
        else:
            sig_price_buy.append(np.nan)
            sig_price_sell.append(np.nan)

    print("Printing from Teacher.py: Profit from these buy/sell signals is {}".format(np.around(budget, 2)))

    # Returning tuples
    return sig_price_buy, sig_price_sell

    # ind_list = list(dataset)
    # matching = [s for s in ind_list if "Ema" in s]
    #
    # sigPriceBuy = []
    # sigPriceSell = []
    # flag = -1
    # for i in range(0, len(dataset)):
    #     # if MACD > signal line  then buy else sell
    #     if dataset[matching[0]][i] < dataset[matching[1]][i] and \
    #             dataset[matching[0]][i] < dataset[matching[2]][i]:
    #         if dataset['Rsi'][i] < 30.0:
    #             if flag != 1:
    #                 sigPriceBuy.append(dataset['Close'][i])
    #                 sigPriceSell.append(np.nan)
    #                 flag = 1
    #             else:
    #                 sigPriceBuy.append(np.nan)
    #                 sigPriceSell.append(np.nan)
    #         else:
    #             sigPriceBuy.append(np.nan)
    #             sigPriceSell.append(np.nan)
    #     elif dataset[matching[0]][i] > dataset[matching[1]][i] and \
    #             dataset[matching[0]][i] > dataset[matching[2]][i]:
    #         if dataset['Rsi'][i] > 70.0:
    #             if flag != 0:
    #                 sigPriceSell.append(dataset['Close'][i])
    #                 sigPriceBuy.append(np.nan)
    #                 flag = 0
    #             else:
    #                 sigPriceBuy.append(np.nan)
    #                 sigPriceSell.append(np.nan)
    #         else:
    #             sigPriceBuy.append(np.nan)
    #             sigPriceSell.append(np.nan)
    #     else:  # Handling nan values
    #         sigPriceBuy.append(np.nan)
    #         sigPriceSell.append(np.nan)
    #
    # return sigPriceBuy, sigPriceSell



