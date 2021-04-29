import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import Modules.services as services
import datetime as dt
import matplotlib.dates as mdates


class OhlcHandler:
    def __init__(self, pair):
        self.pair = pair
        self.interval = None
        self.dataset = None
        self.indicators = []

    def save_to_csv(self, file):
        try:
            self.dataset.to_csv(file)
        except Exception:
            raise ValueError("Something wrong occurred while saving dataset to file!")

    def load_from_csv(self, file):
        try:
            self.dataset = pd.read_csv(file, index_col=0)
            self.dataset.insert(0, 'Date', pd.to_datetime(self.dataset['Date.1']))
            self.dataset.index = self.dataset['Date']
            del self.dataset['Date.1']
            r1 = self.dataset.iloc[0]['Date']
            r2 = self.dataset.iloc[1]['Date']
            interval = r2 - r1
            self.interval = int(interval.total_seconds() / 60.0)
        except Exception:
            raise ValueError("Cannot load dataset from file, file does not exists!")

    def print_to_file(self, file):
        try:
            os.remove(file)
        except OSError:
            pass

        file = open(file, "a")
        if self.dataset is None:
            raise ValueError("Dataset is not created!")
        else:
            file.write(self.dataset.to_string())
            file.write('\n')

        file.close()

    # def plot_to_tkinter(self, ideal_signals: tuple, predicted_signals: tuple, frames, ax: list, canvas: list, fig: list, toolbar):
    #     if len(ax) == 0:
    #         fig1 = plt.figure(figsize=(7, 3))
    #         fig2 = plt.figure(figsize=(7, 3))
    #         fig3 = plt.figure(figsize=(7, 3))
    #         fig4 = plt.figure(figsize=(7, 3))
    #
    #         ax.append(fig1.add_subplot(1, 1, 1))
    #         ax.append(fig2.add_subplot(1, 1, 1))
    #         ax.append(fig3.add_subplot(1, 1, 1))
    #         ax.append(fig4.add_subplot(1, 1, 1))
    #
    #         canvas.append(FigureCanvasTkAgg(fig1, frames[0]))
    #         canvas.append(FigureCanvasTkAgg(fig2, frames[1]))
    #         canvas.append(FigureCanvasTkAgg(fig3, frames[2]))
    #         canvas.append(FigureCanvasTkAgg(fig4, frames[3]))
    #
    #         toolbar1 = tkagg.NavigationToolbar2Tk(canvas[0], frames[0])
    #         toolbar2 = tkagg.NavigationToolbar2Tk(canvas[1], frames[1])
    #         toolbar3 = tkagg.NavigationToolbar2Tk(canvas[2], frames[2])
    #         toolbar4 = tkagg.NavigationToolbar2Tk(canvas[3], frames[3])
    #
    #         toolbar1.config(background='white')
    #         toolbar2.config(background='white')
    #         toolbar3.config(background='white')
    #         toolbar4.config(background='white')
    #
    #         canvas[0].get_tk_widget().pack(side=tk.TOP)
    #         canvas[1].get_tk_widget().pack(side=tk.BOTTOM)
    #         canvas[2].get_tk_widget().pack(side=tk.TOP)
    #         canvas[3].get_tk_widget().pack(side=tk.BOTTOM)
    #
    #         toolbar1.pack()
    #         toolbar2.pack()
    #         toolbar3.pack()
    #         toolbar4.pack()
    #     else:
    #         for a in ax:
    #             a.clear()
    #
    #     if len(predicted_signals[0]) == len(predicted_signals[1]):
    #         if len(set(predicted_signals[0])) != 1:
    #             ax[0].scatter(self.dataset.index, predicted_signals[0], color='green', label='Predicted buy', marker='X', alpha=1)
    #         if len(set(predicted_signals[1])) != 1:
    #             ax[0].scatter(self.dataset.index, predicted_signals[1], color='red', label='Predicted sell', marker='X', alpha=1)
    #     else:
    #         print("\nLength of arrays are not the same, cannot plot!\n")
    #
    #     if len(set(ideal_signals[0])) != 1:
    #         ax[0].scatter(self.dataset.index, ideal_signals[0], color='green', label='Buy Signal', marker='^', alpha=1)
    #     if len(set(ideal_signals[1])) != 1:
    #         ax[0].scatter(self.dataset.index, ideal_signals[1], color='red', label='Sell Signal', marker='v', alpha=1)
    #
    #     ax[0].plot(self.dataset['Close'], 'b', label='Close')
    #     ax[0].grid(True)
    #     ax[0].legend(loc="lower left")
    #
    #     ax[1].plot(self.dataset['Macd'], 'b', label='Macd')  # row=1, col=0
    #     ax[1].plot(self.dataset['Signal'], 'm', label='Signal')  # row=1, col=0
    #     ax[1].plot(self.dataset['Momentum'], 'go', label='Momentum')  # row=1, col=0
    #     ax[1].axhline(y=0, color='k')
    #     ax[1].grid(True)
    #     ax[1].legend(loc="lower right")
    #
    #     ind_list = list(self.dataset)
    #     matching = [s for s in ind_list if "Ema" in s]
    #     ax[2].plot(self.dataset[matching[0]], 'b', label=matching[0])  # row=0, col=0
    #     ax[2].plot(self.dataset[matching[1]], 'm', label=matching[1])  # row=0, col=0
    #     ax[2].grid(True)
    #     ax[2].legend(loc="lower right")
    #
    #     ax[3].plot(self.dataset['Rsi'], 'b', label='Rsi')  # row=0, col=0
    #     ax[3].axhline(y=30, color='r')
    #     ax[3].axhline(y=70, color='r')
    #     ax[3].grid(True)
    #     ax[3].legend(loc="lower right")
    #
    #     canvas[0].draw()
    #     canvas[1].draw()
    #     canvas[2].draw()
    #     canvas[3].draw()
    #
    #     return fig, ax, canvas, toolbar

    def plot_candlestick(self, indicators=False, buy_sell: tuple = None, answers: tuple = None, norm: pd.DataFrame = None, filter_const: int = None):
        if indicators:
            # Plot candlestick chart
            fig = plt.figure()

            if norm is None:
                ax = fig.subplots(nrows=3, ncols=2, sharex=True)
            else:
                ax = fig.subplots(nrows=5, ncols=2, sharex=True)

            ind_list = list(self.dataset)
            matching = [s for s in ind_list if "Ema" in s]
            ax[0, 0].plot(self.dataset[matching[0]], 'b', label=matching[0])  # row=0, col=0
            ax[0, 0].plot(self.dataset[matching[1]], 'm', label=matching[1])  # row=0, col=0
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
            output = services.Filter.fft_filter(input_data, filter_const)
            ax[1, 1].plot(self.dataset.index, output, 'b', label='Filtered close price')  # row=1, col=1
            ax[1, 1].grid(True)
            ax[1, 1].legend(loc="lower right")

            ax[2, 1].plot(self.dataset['Gradient'], 'b', label='Gradient')  # row=1, col=1
            ax[2, 1].grid(True)
            ax[2, 1].legend(loc="lower right")

            if buy_sell is not None:
                if len(set(buy_sell[0])) != 1:
                    ax[0, 1].scatter(self.dataset.index, buy_sell[0], color='green', label='Buy Signal', marker='^', alpha=1)
                if len(set(buy_sell[1])) != 1:
                    ax[0, 1].scatter(self.dataset.index, buy_sell[1], color='red', label='Sell Signal', marker='v', alpha=1)
            if answers is not None:
                if len(set(answers[0])) != 1:
                    ax[0, 1].scatter(self.dataset.index, answers[0], color='green', label='Predicted buy', marker='X', alpha=1)
                if len(set(answers[1])) != 1:
                    ax[0, 1].scatter(self.dataset.index, answers[1], color='red', label='Predicted sell', marker='X', alpha=1)

            ax[0, 1].plot(self.dataset['Close'], 'b', label='Close')  # row=1, col=1
            ax[0, 1].grid(True)
            ax[0, 1].legend(loc="lower right")

            if norm is not None:
                ax[3, 0].plot(self.dataset.index, norm['Norm_Close'], color='b', label='Normalized Close price')
                ax[3, 1].plot(self.dataset.index, norm['Norm_Momentum'], color='b', label='Normalized MACD Momentum')
                ax[3, 0].grid(True)
                ax[3, 0].legend(loc="lower right")
                ax[3, 1].grid(True)
                ax[3, 1].legend(loc="lower right")
                ax[4, 0].plot(self.dataset.index, norm['Norm_Volume'], color='b', label='Normalized volume')
                ax[4, 1].plot(self.dataset.index, norm['Norm_Gradient'], color='b', label='Normalized gradient')
                ax[4, 0].grid(True)
                ax[4, 0].legend(loc="lower right")
                ax[4, 1].grid(True)
                ax[4, 1].legend(loc="lower right")

            for row in ax:
                for a in row:
                    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    a.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        plot_df = self.dataset.iloc[:, :-1]
        mpf.plot(plot_df, type='candle', style='charles', title='Candle Stick Graph', ylabel='Price', volume=True, mav=(3,6,9))

        plt.show()

    def _format_dataframe(self, pd_dtf):
        pd_dtf['Open'] = pd_dtf['Open'].astype('double')
        pd_dtf['High'] = pd_dtf['High'].astype('double')
        pd_dtf['Low'] = pd_dtf['Low'].astype('double')
        pd_dtf['Close'] = pd_dtf['Close'].astype('double')
        pd_dtf['Volume'] = pd_dtf['Volume'].astype('double')
        pd_dtf.index = pd.to_datetime(pd_dtf.Date)
        return pd_dtf

    def _add_statistic_indicators(self, ema_short=3, ema_long=6, macd_slow=12, macd_fast=26, macd_signal=9, rsi_n=6, vwap_n=14):
        if ema_short > ema_long:
            raise Exception("Short ema must be lower than long ema")
        if ema_long < ema_short:
            raise Exception("Long ema must be higher than short ema")

        self.dataset['Close-Open'] = self.dataset['Close'] - self.dataset['Open']
        self.__ema(ema_short)
        self.__ema(ema_long)
        ind_list = list(self.dataset)
        matching = [s for s in ind_list if "Ema" in s]
        self.dataset['{}-{}'.format(matching[0], matching[1])] = self.dataset[matching[0]] - self.dataset[matching[1]]

        self.__macd(macd_slow, macd_fast, macd_signal)
        self.__rsi(rsi_n)
        self.__gradient()

    def __ema(self, n):
        ema_name = 'Ema' + str(n)
        # ema_additional = 'EMA' + str(n) + '_TA'
        self.dataset[ema_name] = self.dataset['Close'].ewm(span=n, adjust=False).mean()
        self.indicators.append(ema_name)

    def __macd(self, nslow, nfast, nsignal):
        macd_name = ['Macd', 'Signal', 'Momentum']
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

    def __vwap(self, n):
        vwap_name = 'Vwap'
        self.dataset[vwap_name] = 0
        v = self.dataset.iloc[:, self.dataset.columns.get_loc('Volume')]
        h = self.dataset.iloc[:, self.dataset.columns.get_loc('High')]
        l = self.dataset.iloc[:, self.dataset.columns.get_loc('Low')]

        vwap = np.cumsum(v * (h + l) / 2) / np.cumsum(v)
        self.dataset[vwap_name] = vwap
        self.indicators.append(vwap_name)

    def __rsi(self, n):
        rsi_name = 'Rsi'
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

    def __gradient(self):
        gradient_name = 'Gradient'
        self.dataset[gradient_name] = [i * 100 for i in np.gradient(self.dataset['Close'])]
        self.indicators.append(gradient_name)

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

