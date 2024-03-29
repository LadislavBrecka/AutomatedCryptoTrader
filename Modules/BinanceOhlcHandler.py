from Modules.OhlcHandler import OhlcHandler
from binance.client import Client
import numpy as np
import pandas as pd
import datetime as dt
from Config import *
from Modules.services import MyLogger


class BinanceOhlcHandler(OhlcHandler):

    # Creating client of BINANCE API with our API_KEY and API_SECRET. It is static variable of this class
    binance_client = Client(API_KEY, API_SECRET)

    def __init__(self, pair):
        # Calling parent constructor
        super(BinanceOhlcHandler, self).__init__(pair)

    # Function for getting actual price of coin for live prediction
    def get_actual_price(self):
        return float(self.binance_client.get_symbol_ticker(symbol=self.pair)['price'])

    # Function for getting dataset into memory for specified number of hours and with specified interval, default is 5m
    def get_dataset(self, hours, interval=INTERVAL):
        if interval == '1m':
            self.interval = 1
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, "{} hours ago UTC".format(hours))
        elif interval == '5m':
            self.interval = 5
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_5MINUTE, "{} hours ago UTC".format(hours))
            pass
        else:
            raise ValueError("Only '1m' or '5m' interval is supported to this day.")

        raw_data = np.array(raw_data)

        self.dataset = pd.DataFrame(raw_data.reshape(-1, 12), dtype=float, columns=('Date',
                                                                        'Open', 'High', 'Low', 'Close',
                                                                        'Volume',
                                                                        'close_time', 'qav',
                                                                        'num_trades',
                                                                        'taker_base_vol',
                                                                        'taker_quote_vol', 'ignore'))

        self.dataset['Date'] = self.dataset['Date'] / 1000
        self.dataset['Date'] = [dt.datetime.fromtimestamp(x) for x in self.dataset['Date']]

        del self.dataset['close_time']
        del self.dataset['qav']
        del self.dataset['num_trades']
        del self.dataset['taker_base_vol']
        del self.dataset['taker_quote_vol']
        del self.dataset['ignore']

        self.dataset = self._format_dataframe(self.dataset)
        self._add_statistic_indicators()

    # Function for getting actual candlestick data
    def get_recent_OHLC(self):
        if self.interval == 1:
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, "1 minute ago UTC")
        elif self.interval == 5:
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_5MINUTE, "5 minutes ago UTC")
        else:
            raw_data = []

        raw_data = np.array(raw_data)

        recent_trade = pd.DataFrame(raw_data.reshape(-1, 12), dtype=float, columns=('Date',
                                                                        'Open', 'High', 'Low', 'Close',
                                                                        'Volume',
                                                                        'close_time', 'qav',
                                                                        'num_trades',
                                                                        'taker_base_vol',
                                                                        'taker_quote_vol', 'ignore'))

        recent_trade['Date'] = recent_trade['Date'] / 1000
        recent_trade['Date'] = [dt.datetime.fromtimestamp(x) for x in recent_trade['Date']]

        del recent_trade['close_time']
        del recent_trade['qav']
        del recent_trade['num_trades']
        del recent_trade['taker_base_vol']
        del recent_trade['taker_quote_vol']
        del recent_trade['ignore']

        recent_trade = self._format_dataframe(recent_trade)

        self.dataset = self.dataset.append(recent_trade, ignore_index=True)
        self.dataset = self._format_dataframe(self.dataset)
        self._add_statistic_indicators()

        MyLogger.write_console("Added new sample for time {} :\n {}".format(self.dataset.iloc[-1]['Date'], self.dataset.iloc[-1]))
