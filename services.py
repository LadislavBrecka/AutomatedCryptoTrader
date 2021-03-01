from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from Constants import *


class Normalize:

    def __init__(self, dataset):
        ind_list = list(dataset)
        matching = [s for s in ind_list if "Ema" in s]

        OPEN_RANGE = [(2.0 - PRICE_AMPLITUDE) * dataset['Open'].min(), PRICE_AMPLITUDE * dataset['Open'].max()]
        HIGH_RANGE = [(2.0 - PRICE_AMPLITUDE) * dataset['High'].min(),  PRICE_AMPLITUDE * dataset['High'].max()]
        LOW_RANGE = [(2.0 - PRICE_AMPLITUDE) * dataset['Low'].min(),  PRICE_AMPLITUDE * dataset['Low'].max()]
        CLOSE_RANGE = [(2.0 - PRICE_AMPLITUDE) * dataset['Close'].min(),  PRICE_AMPLITUDE * dataset['Close'].max()]
        VOLUME_RANGE = [(2 - VOLUME_AMPLITUDE) * dataset['Volume'].min(), VOLUME_AMPLITUDE * dataset['Volume'].max()]
        EMA_SHORT_RANGE = [(2.0 - EMA_AMPLITUDE) * dataset['{}'.format(matching[0])].min(), EMA_AMPLITUDE * dataset['{}'.format(matching[0])].max()]
        EMA_LONG_RANGE = [(2.0 - EMA_AMPLITUDE) * dataset['{}'.format(matching[1])].min(), EMA_AMPLITUDE * dataset['{}'.format(matching[1])].max()]
        DIFF_EMA_RANGE = [DIF_EMA_AMPLITUDE * dataset['{}-{}'.format(matching[0], matching[1])].min(), DIF_EMA_AMPLITUDE * dataset['{}-{}'.format(matching[0], matching[1])].max()]
        MACD_RANGE = [(2.0 - MACD_AMPLITUDE) * dataset['Macd'].min(), MACD_AMPLITUDE * dataset['Macd'].max()]
        SIGNAL_RANGE = [(2.0 - MACD_AMPLITUDE) * dataset['Signal'].min(), MACD_AMPLITUDE * dataset['Signal'].max()]
        MOMENTUM_RANGE = [MOMENTUM_AMPLITUDE * dataset['Momentum'].min(), MOMENTUM_AMPLITUDE * dataset['Momentum'].max()]
        RSI_RANGE = [0.0, 100.0]
        GRADIENT_RANGE = [GRADIENT_AMPLITUDE * dataset['Gradient'].min(), GRADIENT_AMPLITUDE * dataset['Gradient'].max()]

        self.INPUT_RANGES = []
        self.INPUT_RANGES.append(OPEN_RANGE)
        self.INPUT_RANGES.append(HIGH_RANGE)
        self.INPUT_RANGES.append(LOW_RANGE)
        self.INPUT_RANGES.append(CLOSE_RANGE)
        self.INPUT_RANGES.append(VOLUME_RANGE)
        self.INPUT_RANGES.append(EMA_SHORT_RANGE)
        self.INPUT_RANGES.append(EMA_LONG_RANGE)
        self.INPUT_RANGES.append(DIFF_EMA_RANGE)
        self.INPUT_RANGES.append(MACD_RANGE)
        self.INPUT_RANGES.append(SIGNAL_RANGE)
        self.INPUT_RANGES.append(MOMENTUM_RANGE)
        self.INPUT_RANGES.append(RSI_RANGE)
        self.INPUT_RANGES.append(GRADIENT_RANGE)

    # Returning normalized [OPEN, HIGH, LOW, CLOSE, VOLUME, EMA_SHORT, EMA_LONG, EMA_DIFF, MACD, SIGNAL, MOMENTUM, RSI, GRADIENT]
    def get_normalized_row(self, row: list):
        result = []
        for row_item, item_range in zip(row, self.INPUT_RANGES):
            normalized = (row_item - item_range[0]) / (item_range[1] - item_range[0])
            if normalized < 0.0:
                normalized = 0.0
            elif normalized > 1.0:
                normalized = 1.0
            result.append(normalized)

        return result


class Filter:

    @staticmethod
    def fft_filter(input_data: list, filter_const: int) -> list:
        furrier_transform = np.fft.fft(input_data)
        shifted_furrier_transform = np.fft.fftshift(furrier_transform)
        hp_filter = np.zeros(len(shifted_furrier_transform), dtype=int)
        n = int(len(hp_filter))
        hp_filter[int(n / 2) - filter_const: int(n / 2) + filter_const] = 1
        output = shifted_furrier_transform * hp_filter
        output = abs(np.fft.ifft(output))

        return output


class Splitter:

    @staticmethod
    def split_test_train(dataset: pd.DataFrame, test_size: float) -> tuple:
        if test_size > 1.3:
            test_size = 1.3
            raise ValueError("Test size must be max 0.9, setting 0.9 as test size!")
        elif test_size < 0.0:
            test_size = 0.0
            raise ValueError("Test size can not be lower than 0, setting 0.0 as test size!")

        train_length = int((1 - test_size) * len(dataset))

        train = dataset.iloc[0:train_length]
        test = dataset.iloc[train_length:]

        return train, test



