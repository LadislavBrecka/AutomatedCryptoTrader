from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd


class Normalize:

    def __init__(self, dataset):
        ind_list = list(dataset)
        matching = [s for s in ind_list if "Ema" in s]

        OPEN_RANGE = [1.0 * dataset['Open'].min(), 1.0 * dataset['Open'].max()]
        HIGH_RANGE = [1.0 * dataset['High'].min(), 1.0 * dataset['High'].max()]
        LOW_RANGE = [1.0 * dataset['Low'].min(), 1.0 * dataset['Low'].max()]
        CLOSE_RANGE = [1.0 * dataset['Close'].min(), 1.0 * dataset['Close'].max()]
        VOLUME_RANGE = [1.0 * dataset['Volume'].min(), 1.2 * dataset['Volume'].max()]
        DIFF_EMA_RANGE = [0.4 * dataset['{}-{}'.format(matching[0], matching[1])].min(), 0.4 * dataset['{}-{}'.format(matching[0], matching[1])].max()]
        DIFF_MACD_RANGE = [0.4 * dataset['Macd-Signal'].min(), 0.4 * dataset['Macd-Signal'].max()]
        RSI_RANGE = [0.0, 100.0]
        GRADIENT_RANGE = [0.4 * dataset['Gradient'].min(), 0.4 * dataset['Gradient'].max()]

        self.INPUT_RANGES = []
        self.INPUT_RANGES.append(OPEN_RANGE)
        self.INPUT_RANGES.append(HIGH_RANGE)
        self.INPUT_RANGES.append(LOW_RANGE)
        self.INPUT_RANGES.append(CLOSE_RANGE)
        self.INPUT_RANGES.append(VOLUME_RANGE)
        self.INPUT_RANGES.append(DIFF_EMA_RANGE)
        self.INPUT_RANGES.append(DIFF_MACD_RANGE)
        self.INPUT_RANGES.append(RSI_RANGE)
        self.INPUT_RANGES.append(GRADIENT_RANGE)

    # Returning normalized [OPEN, HIGH, LOW, CLOSE, VOLUME, EMA_DIFF, MACD_DIFF, RSI, GRADIENT]
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


class Filter:
    # all_coins = "{'1': 'bitcoin', '1027': 'ethereum', '52': 'xrp', '825': 'tether', '2': 'litecoin', '1831': 'bitcoin-cash', '1975': 'chainlink', '2010': 'cardano', '6636': 'polkadot-new', '1839': 'binance-coin', '512': 'stellar', '3602': 'bitcoin-sv', '3408': 'usd-coin', '1765': 'eos', '328': 'monero', '3717': 'wrapped-bitcoin', '1958': 'tron', '873': 'nem', '2011': 'tezos', '3635': 'crypto-com-coin', '2280': 'filecoin', '3957': 'unus-sed-leo', '1376': 'neo', '3077': 'vechain', '4943': 'multi-collateral-dai', '7278': 'aave', '3794': 'cosmos', '131': 'dash', '2135': 'revain', '1274': 'waves', '7083': 'uniswap', '1720': 'iota', '2502': 'huobi-token', '5864': 'yearn-finance', '1437': 'zcash', '2416': 'theta', '5692': 'compound', '1321': 'ethereum-classic', '4687': 'binance-usd', '1659': 'gnosis-gno', '2700': 'celsius', '2586': 'synthetix-network-token', '1518': 'maker', '1808': 'omg', '5617': 'uma', '2566': 'ontology', '74': 'dogecoin', '5034': 'kusama', '4195': 'ftx-token', '6758': 'sushiswap', '4030': 'algorand', '2469': 'zilliqa', '1697': 'basic-attention-token', '3718': 'bittorrent', '1168': 'decred', '3897': 'okb', '5777': 'renbtc', '2563': 'trueusd', '1896': '0x', '4056': 'ampleforth', '109': 'digibyte', '2539': 'ren', '3330': 'paxos-standard', '1684': 'qtum', '2694': 'nexo', '4642': 'hedera-hashgraph', '4779': 'husd', '2099': 'icon', '5567': 'celo', '3437': 'abbc-coin', '1934': 'loopring', '4847': 'blockstack', '4172': 'terra-luna', '3662': 'hedgetrade', '6535': 'near-protocol', '6892': 'elrond-egld', '1982': 'kyber-network', '5268': 'energy-web-token', '3964': 'reserve-rights', '1104': 'augur', '3911': 'ocean-protocol', '1214': 'lisk', '2083': 'bitcoin-gold', '1042': 'siacoin', '4679': 'band-protocol', '3155': 'quant', '4157': 'thorchain', '1567': 'nano', '1732': 'numeraire', '1966': 'decentraland', '1680': 'aragon', '3351': 'zb-token', '2130': 'enjin-coin', '1759': 'status', '1698': 'horizen', '5026': 'orchid', '291': 'maidsafecoin', '2499': 'swissborg', '2577': 'ravencoin', '1455': 'golem-network-tokens'}"

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
            raise ValueError("Test size must be max 1.3, setting 1.3 as test size!")
        elif test_size < 0.0:
            test_size = 0.0
            raise ValueError("Test size can not be lower than 0, setting 0.0 as test size!")

        train_length = int((1 - test_size) * len(dataset))

        train = dataset.iloc[0:train_length]
        test = dataset.iloc[train_length:]

        return train, test



