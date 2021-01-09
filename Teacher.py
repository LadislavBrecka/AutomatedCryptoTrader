import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from services import Services

# Vygeneruje target list pre vstup trenovania ns
def get_target(look_back: int, look_future: int, dataset: pd.DataFrame) -> list:
    targets = []

    for i in range(1, len(dataset) - look_future, look_back):
        # Logika vyhodnocovania spravneho momentu pre kupu

        current = dataset.iloc[i]
        # step_back = dataset.index[i-1]
        future = dataset.iloc[i + look_future]
        if current['Close'] < future['Close'] + 10.0:
            ans = 0.6
        elif current['Close'] < future['Close'] + 50.0:
            ans = 0.7
        elif current['Close'] < future['Close'] + 100.0:
            ans = 0.85
        elif current['Close'] < future['Close'] + 200.0:
            ans = 1.0
        else:
            ans = 0.01

        targets.append(ans)

    return targets


# Vygeneruje buy/sell signaly podla MACD RSI EMA a nahladnutia do buducnosti
def generate_buy_sell_signal(dataset: pd.DataFrame) -> tuple:
    filtered_price = Services.fft_filter(dataset['Close'], 55)
    peaks, _ = find_peaks(filtered_price, distance=20)
    print(peaks)

    sigPriceBuy = []
    sigPriceSell = []
    flag = -1

    for i in range(0, len(dataset)):
        # if MACD > signal line  then buy else sell
        if i in peaks:
            sigPriceBuy.append(dataset['Close'][i])
            sigPriceSell.append(np.nan)

        elif peaks[1] == -100:
            sigPriceSell.append(dataset['Close'][i])
            sigPriceBuy.append(np.nan)

        else:  # Handling nan values
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return sigPriceBuy, sigPriceSell

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



