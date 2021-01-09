import os
import numpy as np
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from services import Services
import matplotlib.dates as mdates
# pd.options.display.float_format = "{:,.2f}".format


class OhlcHandler:
    def __init__(self, pair):
        self.pair = pair
        self.interval = None
        self.dataset = None
        self.indicators = []

    def print_data(self):
        if self.dataset is None:
            print("Dataset is not created!\n")
        else:
            print(self.dataset)

    def print_to_file(self, file, data='Dataset'):
        try:
            os.remove(file)
        except OSError:
            pass

        file = open(file, "a")
        if self.dataset is None:
            print("Dataset is not created!\n")
        else:
            file.write(self.dataset.to_string())
            file.write('\n')

        file.close()

    def get_closed_price(self):
        if self.dataset is None:
            print("Cannot get closed price, dataset is not created!")
        else:
            closed_price = self.dataset['Close'].astype('float')
            closed_price = closed_price.to_numpy()
            closed_price = closed_price.reshape(len(closed_price), 1)
            return closed_price

    def plot_candlestick(self, indicators=False):
        # Plot candlestick chart
        fig = plt.figure()
        ax = fig.subplots(nrows=3, ncols=2, sharex=True)
        # ax.xaxis_date()
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d, %H:%M'))

        ind_list = list(self.dataset)
        matching = [s for s in ind_list if "Ema" in s]
        ax[0, 0].plot(self.dataset[matching[0]], 'b', label=matching[0])  # row=0, col=0
        ax[0, 0].plot(self.dataset[matching[1]], 'm', label=matching[1])  # row=0, col=0
        ax[0, 0].plot(self.dataset[matching[2]], 'g', label=matching[2])  # row=0, col=0
        ax[0, 0].grid(True)
        ax[0, 0].legend(loc="lower right")

        ax[1, 0].plot(self.dataset['Rsi'], 'b', label='Rsi')  # row=0, col=0
        ax[1, 0].axhline(y=30, color='r')
        ax[1, 0].axhline(y=70, color='r')
        ax[1, 0].grid(True)
        ax[1, 0].legend(loc="lower right")

        ax[2, 0].plot(self.dataset['Macd'], 'b', label='Macd')  # row=1, col=0
        ax[2, 0].plot(self.dataset['Signal'], 'm', label='Signal')  # row=1, col=0
        ax[2, 0].plot(self.dataset['Momentum'], 'go', label='Momentum')  # row=1, col=0
        ax[2, 0].axhline(y=0, color='k')
        ax[2, 0].grid(True)
        ax[2, 0].legend(loc="lower right")

        input_data = self.dataset['Close']
        output = Services.fft_filter(input_data,60)

        ax[1, 1].plot(self.dataset.index, output, 'b', label='Filtered')  # row=1, col=1
        ax[1, 1].grid(True)
        ax[1, 1].legend(loc="lower right")

        ax[2, 1].plot(self.dataset['Gradient'], 'b', label='Gradient')  # row=1, col=1
        ax[2, 1].grid(True)
        ax[2, 1].legend(loc="lower right")

        if 'Buy' in self.dataset:
            ax[0, 1].scatter(self.dataset.index, self.dataset['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
        if 'Sell' in self.dataset:
            ax[0, 1].scatter(self.dataset.index, self.dataset['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
        ax[0, 1].plot(self.dataset['Close'], 'b', label='Close')  # row=1, col=1
        ax[0, 1].grid(True)
        ax[0, 1].legend(loc="lower right")

        # ax.grid(True)
        plt.xticks(rotation=45)

        plot_df = self.dataset.iloc[:, :-1]
        mpf.plot(plot_df, type='candle', style='charles', title='Candle Stick Graph', ylabel='Price', volume=True, mav=(3,6,9))


        plt.show()

    def __insert_row(self, row_number, row_value):
        # Starting value of upper half
        start_upper = 0
        # End value of upper half
        end_upper = row_number
        # Start value of lower half
        start_lower = row_number
        # End value of lower half
        end_lower = self.dataset.shape[0]
        # Create a list of upper_half index
        upper_half = [*range(start_upper, end_upper, 1)]
        # Create a list of lower_half index
        lower_half = [*range(start_lower, end_lower, 1)]
        # Increment the value of lower half by 1
        lower_half = [x.__add__(1) for x in lower_half]
        # Combine the two lists
        index_ = upper_half + lower_half
        # Update the index of the dataframe
        self.dataset.index = index_
        # Insert a row at the end
        self.dataset.loc[row_number] = row_value
        # Sort the index labels
        self.dataset = self.dataset.sort_index()

    def _fill_with_nan(self):
        one_minute = dt.timedelta(minutes=1)

        if self.interval == '1':
            n = self.dataset['Date'].iloc[0]
            n = n - one_minute
            n = int(n.strftime("%M"))

            for d in self.dataset['Date']:
                prev_n = n
                n = int(d.strftime("%M"))
                if n < prev_n:
                    prev_n = prev_n - 60
                dif = n - prev_n

                if (dif > 1) and not (n == 0 and prev_n == 59):
                    i = self.dataset[self.dataset['Date'] == d].index.values.astype(int)[0]
                    for j in range(dif - 1):
                        d = d - one_minute
                        self.__insert_row(i, [d, np.nan, np.nan, np.nan, np.nan, np.nan])

        elif self.interval == '60':
            pass

    def _format_dataframe(self, pd_dtf):
        pd_dtf['Open'] = pd_dtf['Open'].astype('double')
        pd_dtf['High'] = pd_dtf['High'].astype('double')
        pd_dtf['Low'] = pd_dtf['Low'].astype('double')
        pd_dtf['Close'] = pd_dtf['Close'].astype('double')
        pd_dtf['Volume'] = pd_dtf['Volume'].astype('double')
        pd_dtf.index = pd.to_datetime(pd_dtf.Date)
        # self.dataset.drop('Date', axis=1, inplace=True)
        # cut = self.dataset.shape[0] - 500
        # self.dataset = self.dataset.iloc[cut:]
        # self.dataset.index = (i for i in range(500))
        return pd_dtf

    def _add_statistic_indicators(self):
        self.__ema(3)
        self.__ema(6)
        self.__ema(9)
        self.__macd()
        self.__rsi()
        self.__vwap()
        self.__gradient()

    def __ema(self, n=7):
        ema_name = 'Ema' + str(n)
        # ema_additional = 'EMA' + str(n) + '_TA'
        self.dataset[ema_name] = self.dataset['Close'].ewm(span=n, adjust=False).mean()
        self.indicators.append(ema_name)

        # Ta-lib version
        # self.dataset[ema_additional] = talib.EMA(self.dataset['Close'].values, timeperiod=n)
        # self.indicators.append(ema_additional)

    def __macd(self, nslow=12, nfast=26, nsignal=9):
        macd_name = ['Macd', 'Signal', 'Momentum']
        # macd_additional = 'MACD_TA'
        closed = self.dataset['Close']
        ema_slw = closed.ewm(span=nslow, min_periods=1, adjust=False).mean()
        ema_fst = closed.ewm(span=nfast, min_periods=1, adjust=False).mean()
        macd = ema_slw - ema_fst
        signal = macd.ewm(span=nsignal, min_periods=1, adjust=False).mean()
        momentum = macd - signal
        self.dataset[macd_name[0]] = macd
        self.dataset[macd_name[1]] = signal
        self.dataset[macd_name[2]] = momentum
        self.indicators.append(macd_name[0])

        # Ta-lib version
        # self.dataset[macd_additional], a, b = talib.MACD(self.dataset["Close"].values, fastperiod=nslow, slowperiod=nfast, signalperiod=nsignal)
        # self.indicators.append(macd_additional)

    # def __vwap(self, n=14):
    #     vwap_name = 'VWAP'
    #     self.dataset[vwap_name] = 0
    #     for i in range(0, 720, n):
    #         tmp_df = self.dataset.iloc[i:i+n]
    #         cum_vol = tmp_df['Volume'].cumsum()
    #         cum_vol_price = (tmp_df['Volume'] * (tmp_df['High'] + tmp_df['Low'] + tmp_df['Close']) / 3.0).cumsum()
    #         self.dataset.iloc[i:i+n, self.dataset.columns.get_loc('VWAP')] = cum_vol_price / cum_vol
    #     self.indicators.append(vwap_name)

    def __vwap(self, n=14):
        vwap_name = 'Vwap'
        self.dataset[vwap_name] = 0
        v = self.dataset.iloc[:, self.dataset.columns.get_loc('Volume')]
        h = self.dataset.iloc[:, self.dataset.columns.get_loc('High')]
        l = self.dataset.iloc[:, self.dataset.columns.get_loc('Low')]

        vwap = np.cumsum(v * (h + l) / 2) / np.cumsum(v)
        self.dataset[vwap_name] = vwap
        self.indicators.append(vwap_name)

    def __rsi(self, n=6):
        rsi_name = 'Rsi'
        # rsi_additional = 'RSI_TA'
        closed = self.dataset['Close']
        deltas = np.diff(closed)
        seed = deltas[:n + 1]
        up = seed[seed >= 0].sum() / n
        down = -seed[seed < 0].sum() / n
        rs = up / down
        rsi = np.zeros_like(closed)
        rsi[:n] = 100. - 100. / (1. + rs)

        for i in range(n, len(closed)):
            delta = deltas[i - 1]  # cause the diff is 1 shorter
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            up = (up * (n - 1) + upval) / n
            down = (down * (n - 1) + downval) / n
            rs = up / down
            rsi[i] = 100. - 100. / (1. + rs)
        self.dataset[rsi_name] = rsi
        self.indicators.append(rsi_name)

        # Ta-lib version
        # self.dataset[rsi_additional] = talib.RSI(self.dataset["Close"].values, timeperiod=6)
        # self.indicators.append(rsi_additional)

    def __gradient(self):
        gradient_name = 'Gradient'
        self.dataset[gradient_name] = [i * 100 for i in np.gradient(self.dataset['Close'])]
        # self.dataset[gradient_name] = np.gradient(self.dataset['Close'], 60 * self.interval)*100
        self.indicators.append(gradient_name)

    # USER CAN CHOOSE WHAT WILL BE PLOTED
    # def plot_closed_price(self, data='All'):
    #     # Graph stuff, axes renaming etc..
    #     ax = plt.gca()
    #     ax.set_ylabel("Price in USD")
    #     ax.set_title("STOCK PRICE EVOLUTION")
    #
    #     if self.closed_price is None:
    #         print("Cannot plot, closed price wasnt generated!")
    #
    #     else:
    #         if data == 'All':
    #             # self.closed_price.plot()
    #             plt.plot(self.closed_price)
    #
    #         # if data is 'Train':
    #         #     if self.train is None:
    #         #         print("Train dataset is not created!\n")
    #         #     else:
    #         #         x = [dt.datetime.strptime(self.dates[i], '%Y-%m-%d').date() for i in range(len(self.train))]
    #         #         plt.plot(x, self.train, color='red', label='Train Dataset')
    #         #
    #         # if data is 'Test':
    #         #     if self.test is None:
    #         #         print("Test dataset is not created!\n")
    #         #     else:
    #         #         x = [dt.datetime.strptime(self.dates[i], '%Y-%m-%d').date() for i in
    #         #              range(len(self.train), len(self.closedADJ_price))]
    #         #         plt.plot(x, self.test, color='blue', label='Test Dataset')
    #         #
    #         # if data is 'Combined':
    #         #     if self.train is None or self.test is None:
    #         #         print("Train and test dataset is not created!\n")
    #         #     else:
    #         #         x1 = [dt.datetime.strptime(self.dates[i], '%Y-%m-%d').date() for i in range(len(self.train))]
    #         #         x2 = [dt.datetime.strptime(self.dates[i], '%Y-%m-%d').date() for i in
    #         #               range(len(self.train), len(self.closedADJ_price))]
    #         #         plt.plot(x1, self.train, color='red')
    #         #         plt.plot(x2, self.test, color='blue')
    #
    #         maximum = max(self.closed_price)
    #         minimum = min(self.closed_price)
    #         difference = maximum - minimum
    #         plt.ylim(minimum - difference / 10, maximum + difference / 10)
    #         plt.show()

