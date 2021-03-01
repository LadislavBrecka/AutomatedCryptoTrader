from OhlcHandler import OhlcHandler
from binance.client import Client
import numpy as np
import pandas as pd
import datetime as dt

api_key = 'wzDZXGFp9DK9QOChVTEOhZKaYEVAbbCBhsBDJXvSI78t2WlD3TqQhViWb8LH5Xom'
api_secret = 'U6Gk59emVhyiNtuX9GDkpgtRwZlsAJkVhUFFzb3YvJEnN8NRDECozDnE8SgyT0kh'


class BinanceOhlcHandler(OhlcHandler):
    #####
    binance_client = Client(api_key, api_secret)
    #####

    def __init__(self, pair, interval='1m'):
        super(BinanceOhlcHandler, self).__init__(pair)

        if interval == '1m':
            self.interval = '1'
        elif interval == '5m':
            self.interval = '5'
        else:
            raise ValueError("Only '1m' or '1h' interval is supported to this day.")

    def get_dataset(self):
        if self.interval == '1':
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, "12 hours ago UTC")
        elif self.interval == '5':
            raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_5MINUTE, "48 hours ago UTC")
        else:
            raise ValueError("Only '1m' or '1h' interval is supported to this day.")

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

    def get_recent_OHLC(self):

        raw_data = self.binance_client.get_historical_klines(self.pair, Client.KLINE_INTERVAL_1MINUTE, "1 minute ago UTC")

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























    # def __make_and_format_binance_dataset(self, data):
    #     pdt = pd.DataFrame(data.reshape(-1, 12), dtype=float, columns=('Date',
    #                                                                     'Open', 'High', 'Low', 'Close',
    #                                                                     'Volume',
    #                                                                     'close_time', 'qav',
    #                                                                     'num_trades',
    #                                                                     'taker_base_vol',
    #                                                                     'taker_quote_vol', 'ignore'))
    #
    #     pdt['Date'] = pdt['Date'] / 1000
    #     pdt['Date'] = [dt.datetime.fromtimestamp(x) for x in pdt['Date']]
    #
    #     del pdt['close_time']
    #     del pdt['qav']
    #     del pdt['num_trades']
    #     del pdt['taker_base_vol']
    #     del pdt['taker_quote_vol']
    #     del pdt['ignore']
    #
    #     # self._format_dataset()
    #     self._add_statistic_indicators(pdt)
    #
    #     return pdt