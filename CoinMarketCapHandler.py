from bs4 import BeautifulSoup
import requests
import pandas as pd
import json
import time


class CoinMarketCapHandler:
    def __init__(self, coin=None):
        self.coin = coin
        self.actual = None
        self.historical = None

    def select_coin(self, coins):
        print(coins)
        print("Select coin in above dict\n")
        self.coin = input()

    def get_actual(self):
        coins = {}
        cmc = requests.get("https://coinmarketcap.com/")
        soup = BeautifulSoup(cmc.content, "html.parser")

        data = soup.find("script", id="__NEXT_DATA__", type="application/json")

        raw_coin_data = json.loads(data.contents[0])
        all_coins = raw_coin_data["props"]["initialState"]["cryptocurrency"][
            "listingLatest"
        ]["data"]

        for i in all_coins:
            coins[str(i["id"])] = i["slug"]

        if self.coin is None:
            self.select_coin(coins)

        my_coin = next(item for item in all_coins if item["slug"] == self.coin)

        timestamp = []
        slug = []
        price = []
        volume_24h = []
        percent_change_1h = []
        percent_change_24h = []

        timestamp.append(my_coin["last_updated"])
        slug.append(my_coin["slug"])
        price.append(my_coin["quote"]["USD"]["price"])
        volume_24h.append(my_coin["quote"]["USD"]["volume_24h"])
        percent_change_1h.append(my_coin["quote"]["USD"]["percent_change_1h"])
        percent_change_24h.append(my_coin["quote"]["USD"]["percent_change_24h"])

        df = pd.DataFrame(
            columns=[
                "timestamp",
                "slug",
                "price",
                "24h_volume",
                "percent_change_1h",
                "percent_change_24h",
            ]
        )

        df["timestamp"] = timestamp
        df["slug"] = slug
        df["price"] = price
        df["24h_volume"] = volume_24h
        df["percent_change_1h"] = percent_change_1h
        df["percent_change_24h"] = percent_change_24h

        self.actual = df

    def get_historic(self):
        coins = {}
        cmc = requests.get("https://coinmarketcap.com/")
        soup = BeautifulSoup(cmc.content, "html.parser")

        data = soup.find("script", id="__NEXT_DATA__", type="application/json")

        raw_coin_data = json.loads(data.contents[0])
        all_coins = raw_coin_data["props"]["initialState"]["cryptocurrency"][
            "listingLatest"
        ]["data"]

        for i in all_coins:
            coins[str(i["id"])] = i["slug"]

        if self.coin is None:
            self.select_coin(coins)

        print("Getting also historical data, wait!")
        print(self.coin)

        key = [k for k, v in coins.items() if v == self.coin]
        key = key[0]

        url = f"https://coinmarketcap.com/currencies/{self.coin}/historical-data/?start=20201206&end=20201208"
        print(url)
        page = requests.get(url)

        soup = BeautifulSoup(page.content, "html.parser")
        data = soup.find("script", id="__NEXT_DATA__", type="application/json")
        historical_data = json.loads(data.contents[0])

        quotes = historical_data["props"]["initialState"]["cryptocurrency"][
            "ohlcvHistorical"
        ][key]["quotes"]
        info = historical_data["props"]["initialState"]["cryptocurrency"][
            "ohlcvHistorical"
        ][key]

        market_cap = []
        volume = []
        timestamp = []
        name = []
        symbol = []
        slug = []
        open = []
        high = []
        low = []
        close = []

        for j in quotes:
            timestamp.append(j["quote"]["USD"]["timestamp"])
            market_cap.append(j["quote"]["USD"]["market_cap"])
            volume.append(j["quote"]["USD"]["volume"])
            open.append(j["quote"]["USD"]["open"])
            high.append(j["quote"]["USD"]["high"])
            low.append(j["quote"]["USD"]["low"])
            close.append(j["quote"]["USD"]["close"])
            name.append(info["name"])
            symbol.append(info["symbol"])
            slug.append(self.coin)

        df = pd.DataFrame(
            columns=[
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "marketcap",
                "name",
                "symbol",
                "slug",
            ]
        )

        df["timestamp"] = timestamp
        df["open"] = open
        df["high"] = high
        df["low"] = low
        df["close"] = close
        df["volume"] = volume
        df["marketcap"] = market_cap
        df["name"] = name
        df["symbol"] = symbol
        df["slug"] = slug

        self.historical = df
