from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from Constants import *
import sys


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

    def get_normalized_dataset(self, dataset):
        '''
        Normalizing whole dataset once
        '''
        temp_open = []
        temp_high = []
        temp_low = []
        temp_close = []
        temp_volume = []
        temp_emaShort = []
        temp_emaLong = []
        temp_ema_diff = []
        temp_macd = []
        temp_signal = []
        temp_momentum = []
        temp_rsi = []
        temp_gradient = []
        for i in range(len(dataset)):
            row = dataset.iloc[i]
            del row['Date']
            row = row.values
            norm_row = self.get_normalized_row(row)
            temp_open.append(norm_row[0])  # open
            temp_high.append(norm_row[1])  # high
            temp_low.append(norm_row[2])  # low
            temp_close.append(norm_row[3])  # close
            temp_volume.append(norm_row[4])  # volume
            temp_emaShort.append(norm_row[5])  # ema short
            temp_emaLong.append(norm_row[6])  # ema long
            temp_ema_diff.append(norm_row[7])  # ema_short - ema_long
            temp_macd.append(norm_row[8])  # macd
            temp_signal.append(norm_row[9])  # signal
            temp_momentum.append(norm_row[10])  # momentum
            temp_rsi.append(norm_row[11])  # rsi
            temp_gradient.append(norm_row[12])  # gradient

        norm_dataset = pd.DataFrame(index=dataset['Date'],
                                    columns=['Norm_Open', 'Norm_High', 'Norm_Low', 'Norm_Close', 'Norm_Volume',
                                             'Norm_EmaShort', 'Norm_EmaLong', 'Norm_EmaDiff', 'Norm_Macd',
                                             'Norm_Signal', 'Norm_Momentum', 'Norm_Rsi', 'Norm_Gradient'])

        norm_dataset['Norm_Open'] = temp_open
        norm_dataset['Norm_High'] = temp_high
        norm_dataset['Norm_Low'] = temp_low
        norm_dataset['Norm_Close'] = temp_close
        norm_dataset['Norm_Volume'] = temp_volume
        norm_dataset['Norm_EmaShort'] = temp_emaShort
        norm_dataset['Norm_EmaLong'] = temp_emaLong
        norm_dataset['Norm_EmaDiff'] = temp_ema_diff
        norm_dataset['Norm_Macd'] = temp_macd
        norm_dataset['Norm_Signal'] = temp_signal
        norm_dataset['Norm_Momentum'] = temp_momentum
        norm_dataset['Norm_Rsi'] = temp_rsi
        norm_dataset['Norm_Gradient'] = temp_gradient

        return norm_dataset


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


class MyLogger():

    @staticmethod
    def write(msg: str, file):
        print(msg)
        file.write(msg + "\n")

    @staticmethod
    def write_console(msg: str):
        print(msg)

    @staticmethod
    def write_file(msg: str, file):
        file.write(msg + "\n")

    # Print iterations progress
    @staticmethod
    def progressBar(iterable, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd=""):
        """
        Call in a loop to create terminal progress bar
        @params:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            prefix      - Optional  : prefix string (Str)
            suffix      - Optional  : suffix string (Str)
            decimals    - Optional  : positive number of decimals in percent complete (Int)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
            printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        """
        total = len(iterable)

        # Progress Bar Printing Function
        def printProgressBar(iteration):
            percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
            filledLength = int(length * iteration // total)
            bar = fill * filledLength + '-' * (length - filledLength)
            print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)

        # Initial Call
        printProgressBar(0)
        # Update Progress Bar
        for i, item in enumerate(iterable):
            yield item
            printProgressBar(i + 1)
        # Print New Line on Complete
        print()




