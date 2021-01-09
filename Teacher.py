import pandas as pd
import numpy as np

# Vygeneruje target list pre vstup trenovania ns
def get_target(look_back: int, look_future: int, dataset: pd.DataFrame) -> list:
    targets = []

    for i in range(1, len(dataset)-look_future, look_back):
        # Logika vyhodnocovania spravneho momentu pre kupu

        current = dataset.iloc[i]
        # step_back = dataset.index[i-1]
        future = dataset.iloc[i+look_future]
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
    sigPriceBuy = []
    sigPriceSell = []
    flag = -1
    for i in range(0, len(dataset)):
        # if MACD > signal line  then buy else sell
        if dataset['Macd'][i] > dataset['Signal'][i]:
            if dataset['Rsi'][i] < 35.0:
                if flag != 1:
                    sigPriceBuy.append(dataset['Close'][i])
                    sigPriceSell.append(np.nan)
                    flag = 1
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        elif dataset['Macd'][i] < dataset['Signal'][i]:
            if dataset['Rsi'][i] > 65.0:
                if flag != 0:
                    sigPriceSell.append(dataset['Close'][i])
                    sigPriceBuy.append(np.nan)
                    flag = 0
                else:
                    sigPriceBuy.append(np.nan)
                    sigPriceSell.append(np.nan)
            else:
                sigPriceBuy.append(np.nan)
                sigPriceSell.append(np.nan)
        else:  # Handling nan values
            sigPriceBuy.append(np.nan)
            sigPriceSell.append(np.nan)

    return sigPriceBuy, sigPriceSell
