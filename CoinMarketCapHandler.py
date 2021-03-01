from bs4 import BeautifulSoup
import requests
import pandas as pd
import json


class CoinMarketCapHandler:
    # all_coins = "{'1': 'bitcoin', '1027': 'ethereum', '52': 'xrp', '825': 'tether', '2': 'litecoin', '1831': 'bitcoin-cash', '1975': 'chainlink', '2010': 'cardano', '6636': 'polkadot-new', '1839': 'binance-coin', '512': 'stellar', '3602': 'bitcoin-sv', '3408': 'usd-coin', '1765': 'eos', '328': 'monero', '3717': 'wrapped-bitcoin', '1958': 'tron', '873': 'nem', '2011': 'tezos', '3635': 'crypto-com-coin', '2280': 'filecoin', '3957': 'unus-sed-leo', '1376': 'neo', '3077': 'vechain', '4943': 'multi-collateral-dai', '7278': 'aave', '3794': 'cosmos', '131': 'dash', '2135': 'revain', '1274': 'waves', '7083': 'uniswap', '1720': 'iota', '2502': 'huobi-token', '5864': 'yearn-finance', '1437': 'zcash', '2416': 'theta', '5692': 'compound', '1321': 'ethereum-classic', '4687': 'binance-usd', '1659': 'gnosis-gno', '2700': 'celsius', '2586': 'synthetix-network-token', '1518': 'maker', '1808': 'omg', '5617': 'uma', '2566': 'ontology', '74': 'dogecoin', '5034': 'kusama', '4195': 'ftx-token', '6758': 'sushiswap', '4030': 'algorand', '2469': 'zilliqa', '1697': 'basic-attention-token', '3718': 'bittorrent', '1168': 'decred', '3897': 'okb', '5777': 'renbtc', '2563': 'trueusd', '1896': '0x', '4056': 'ampleforth', '109': 'digibyte', '2539': 'ren', '3330': 'paxos-standard', '1684': 'qtum', '2694': 'nexo', '4642': 'hedera-hashgraph', '4779': 'husd', '2099': 'icon', '5567': 'celo', '3437': 'abbc-coin', '1934': 'loopring', '4847': 'blockstack', '4172': 'terra-luna', '3662': 'hedgetrade', '6535': 'near-protocol', '6892': 'elrond-egld', '1982': 'kyber-network', '5268': 'energy-web-token', '3964': 'reserve-rights', '1104': 'augur', '3911': 'ocean-protocol', '1214': 'lisk', '2083': 'bitcoin-gold', '1042': 'siacoin', '4679': 'band-protocol', '3155': 'quant', '4157': 'thorchain', '1567': 'nano', '1732': 'numeraire', '1966': 'decentraland', '1680': 'aragon', '3351': 'zb-token', '2130': 'enjin-coin', '1759': 'status', '1698': 'horizen', '5026': 'orchid', '291': 'maidsafecoin', '2499': 'swissborg', '2577': 'ravencoin', '1455': 'golem-network-tokens'}"
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
