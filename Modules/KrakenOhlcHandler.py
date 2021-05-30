from OhlcHandler import OhlcHandler
import requests
import json
import pandas as pd
import datetime as dt
import time


# Not used in project
class KrakenOhlcHandler(OhlcHandler):

    def __init__(self, pair, interval='1m', since='12h'):
        # Calling parent constructor
        super(KrakenOhlcHandler, self).__init__(pair)

        self.since = time.time() - int(since[:-1]) * 60 * 60

        if interval == '1m':
            self.interval = '1'
        elif interval == '1h':
            self.interval = '60'
        else:
            raise ValueError("Only '1m' or '1h' interval is supported to this day.")

        root_url = 'https://api.kraken.com/0/public/OHLC'
        self.url = root_url + '?pair=' + self.pair + '&interval=' + self.interval + '&since=' + str(self.since)

    # Function for getting actual price of coin for live prediction
    def get_dataset(self):
        raw_data = json.loads(requests.get(self.url).text)

        raw_list = []

        for d in raw_data['result'][self.pair]:
            raw_list.append(d)
            self.dataset = pd.DataFrame(raw_list)
            self.dataset.columns = ['Date',
                                    'Open', 'High', 'Low', 'Close', 'Vwap',
                                    'Volume', 'Count']

        self.dataset['Date'] = [dt.datetime.fromtimestamp(x) for x in self.dataset['Date']]

        del self.dataset['Vwap']
        del self.dataset['Count']
        self._fill_with_nan()

        self.dataset = self._format_dataframe(self.dataset)
        self._add_statistic_indicators()

    # Function for getting dataset into memory for specified number of hours and with specified interval, default is 5m
    def get_recent_OHLC(self):
        root_url = 'https://api.kraken.com/0/public/OHLC'
        url = root_url + '?pair=' + self.pair + '&interval=' + self.interval + '&since=' + str(time.time())

        raw_data = json.loads(requests.get(url).text)

        raw_list = []

        recent_trade = pd.DataFrame()
        for d in raw_data['result'][self.pair]:
            raw_list.append(d)
            recent_trade = pd.DataFrame(raw_list)
            recent_trade.columns = ['Date',
                                    'Open', 'High', 'Low', 'Close', 'Vwap',
                                    'Volume', 'Count']

        recent_trade['Date'] = [dt.datetime.fromtimestamp(x) for x in recent_trade['Date']]

        del recent_trade['Vwap']
        del recent_trade['Count']

        recent_trade = self._format_dataframe(recent_trade)
        self.dataset = self.dataset.append(recent_trade, ignore_index=True)
        self.dataset = self._format_dataframe(self.dataset)
        self._add_statistic_indicators()

