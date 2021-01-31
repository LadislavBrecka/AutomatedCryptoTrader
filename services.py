from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class Services:
    all_coins = "{'1': 'bitcoin', '1027': 'ethereum', '52': 'xrp', '825': 'tether', '2': 'litecoin', '1831': 'bitcoin-cash', '1975': 'chainlink', '2010': 'cardano', '6636': 'polkadot-new', '1839': 'binance-coin', '512': 'stellar', '3602': 'bitcoin-sv', '3408': 'usd-coin', '1765': 'eos', '328': 'monero', '3717': 'wrapped-bitcoin', '1958': 'tron', '873': 'nem', '2011': 'tezos', '3635': 'crypto-com-coin', '2280': 'filecoin', '3957': 'unus-sed-leo', '1376': 'neo', '3077': 'vechain', '4943': 'multi-collateral-dai', '7278': 'aave', '3794': 'cosmos', '131': 'dash', '2135': 'revain', '1274': 'waves', '7083': 'uniswap', '1720': 'iota', '2502': 'huobi-token', '5864': 'yearn-finance', '1437': 'zcash', '2416': 'theta', '5692': 'compound', '1321': 'ethereum-classic', '4687': 'binance-usd', '1659': 'gnosis-gno', '2700': 'celsius', '2586': 'synthetix-network-token', '1518': 'maker', '1808': 'omg', '5617': 'uma', '2566': 'ontology', '74': 'dogecoin', '5034': 'kusama', '4195': 'ftx-token', '6758': 'sushiswap', '4030': 'algorand', '2469': 'zilliqa', '1697': 'basic-attention-token', '3718': 'bittorrent', '1168': 'decred', '3897': 'okb', '5777': 'renbtc', '2563': 'trueusd', '1896': '0x', '4056': 'ampleforth', '109': 'digibyte', '2539': 'ren', '3330': 'paxos-standard', '1684': 'qtum', '2694': 'nexo', '4642': 'hedera-hashgraph', '4779': 'husd', '2099': 'icon', '5567': 'celo', '3437': 'abbc-coin', '1934': 'loopring', '4847': 'blockstack', '4172': 'terra-luna', '3662': 'hedgetrade', '6535': 'near-protocol', '6892': 'elrond-egld', '1982': 'kyber-network', '5268': 'energy-web-token', '3964': 'reserve-rights', '1104': 'augur', '3911': 'ocean-protocol', '1214': 'lisk', '2083': 'bitcoin-gold', '1042': 'siacoin', '4679': 'band-protocol', '3155': 'quant', '4157': 'thorchain', '1567': 'nano', '1732': 'numeraire', '1966': 'decentraland', '1680': 'aragon', '3351': 'zb-token', '2130': 'enjin-coin', '1759': 'status', '1698': 'horizen', '5026': 'orchid', '291': 'maidsafecoin', '2499': 'swissborg', '2577': 'ravencoin', '1455': 'golem-network-tokens'}"

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    # TODO
    # Column based normalisation with possible different future min/max
    def normalize(self, dataset: list) -> list:
        return self.scaler.fit_transform(dataset)
        # x = dataset.values  # returns a numpy array
        # x_scaled = self.scaler.fit_transform(x)
        # return pd.DataFrame(x_scaled)

    def unnormalize(self, data_to_unnormalize: list) -> list:
        return self.scaler.inverse_transform(data_to_unnormalize)

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

    @staticmethod
    def split_test_train(dataset: pd.DataFrame, test_size: float) -> tuple:
        if test_size > 0.8:
            test_size = 0.8
            raise ValueError("Test size must be max 0.8, setting 0.8 as test size!")
        elif test_size < 0.0:
            test_size = 0.0
            raise ValueError("Test size can not be lower than 0, setting 0.0 as test size!")

        train_length = int((1 - test_size) * len(dataset))

        train = dataset.iloc[0:train_length]
        test = dataset.iloc[train_length:]

        return train, test



