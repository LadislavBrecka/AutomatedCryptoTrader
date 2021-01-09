from sklearn.preprocessing import MinMaxScaler
from BinanceOhlcHandler import BinanceOhlcHandler
from CoinMarketCapHandler import CoinMarketCapHandler
from OhlcHandler import OhlcHandler
import pandas as pd
import numpy as np


class Services:
    all_coins = "{'1': 'bitcoin', '1027': 'ethereum', '52': 'xrp', '825': 'tether', '2': 'litecoin', '1831': 'bitcoin-cash', '1975': 'chainlink', '2010': 'cardano', '6636': 'polkadot-new', '1839': 'binance-coin', '512': 'stellar', '3602': 'bitcoin-sv', '3408': 'usd-coin', '1765': 'eos', '328': 'monero', '3717': 'wrapped-bitcoin', '1958': 'tron', '873': 'nem', '2011': 'tezos', '3635': 'crypto-com-coin', '2280': 'filecoin', '3957': 'unus-sed-leo', '1376': 'neo', '3077': 'vechain', '4943': 'multi-collateral-dai', '7278': 'aave', '3794': 'cosmos', '131': 'dash', '2135': 'revain', '1274': 'waves', '7083': 'uniswap', '1720': 'iota', '2502': 'huobi-token', '5864': 'yearn-finance', '1437': 'zcash', '2416': 'theta', '5692': 'compound', '1321': 'ethereum-classic', '4687': 'binance-usd', '1659': 'gnosis-gno', '2700': 'celsius', '2586': 'synthetix-network-token', '1518': 'maker', '1808': 'omg', '5617': 'uma', '2566': 'ontology', '74': 'dogecoin', '5034': 'kusama', '4195': 'ftx-token', '6758': 'sushiswap', '4030': 'algorand', '2469': 'zilliqa', '1697': 'basic-attention-token', '3718': 'bittorrent', '1168': 'decred', '3897': 'okb', '5777': 'renbtc', '2563': 'trueusd', '1896': '0x', '4056': 'ampleforth', '109': 'digibyte', '2539': 'ren', '3330': 'paxos-standard', '1684': 'qtum', '2694': 'nexo', '4642': 'hedera-hashgraph', '4779': 'husd', '2099': 'icon', '5567': 'celo', '3437': 'abbc-coin', '1934': 'loopring', '4847': 'blockstack', '4172': 'terra-luna', '3662': 'hedgetrade', '6535': 'near-protocol', '6892': 'elrond-egld', '1982': 'kyber-network', '5268': 'energy-web-token', '3964': 'reserve-rights', '1104': 'augur', '3911': 'ocean-protocol', '1214': 'lisk', '2083': 'bitcoin-gold', '1042': 'siacoin', '4679': 'band-protocol', '3155': 'quant', '4157': 'thorchain', '1567': 'nano', '1732': 'numeraire', '1966': 'decentraland', '1680': 'aragon', '3351': 'zb-token', '2130': 'enjin-coin', '1759': 'status', '1698': 'horizen', '5026': 'orchid', '291': 'maidsafecoin', '2499': 'swissborg', '2577': 'ravencoin', '1455': 'golem-network-tokens'}"

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def normalize(self, data_to_normalize):
        return self.scaler.fit_transform(data_to_normalize)

    def unnormalize(self, data_to_unnormalize):
        return self.scaler.inverse_transform(data_to_unnormalize)
    #
    # def get_data_for_ui(self, dataset, coinmarket):
    #
    #     df_list = dataset.values.tolist()
    #
    #     [lists.pop(0) for lists in df_list]
    #
    #     inputs = self.normalize(df_list)
    #
    #     return inputs


