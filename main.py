from BinanceOhlcHandler import BinanceOhlcHandler
from services import Services
from CoinMarketCapHandler import CoinMarketCapHandler
import Teacher
import numpy as np
import NeuralNetwork_2_hidden as nn_2_hidden
import NeuralNetwork_2_hidden as nn_1_hidden


'''
Setting and getting dataset
'''
binance_pair = 'BTCUSDT'
coin_name = 'bitcoin'

nn = nn_2_hidden.NeuralNetwork(28, 10, 50, 3, 0.3)
# nn = nn_1_hidden.NeuralNetwork(24, 50, 1, 0.3)

services = Services()

binance = BinanceOhlcHandler(binance_pair)
binance.get_dataset()
coinmarket = CoinMarketCapHandler(coin_name)
coinmarket.get_actual()
print(coinmarket.actual)

look_back = 2
epochs = 10
filter_const = 100      # 100 alebo 25, podla toho kolko moc operacii chceme vykonat

'''
Finding and setting buy/sell signals
'''
x = Teacher.generate_buy_sell_signal(binance.dataset, filter_const)
targets = Teacher.get_target(binance.dataset, x)

binance.dataset['Buy'] = x[0]
binance.dataset['Sell'] = x[1]
binance.dataset['Target'] = targets
'''
Plotting and printing
'''

binance.plot_candlestick(indicators=True, buy_sell=x, filter_const=filter_const)

binance.print_to_file('out1.txt')

'''
Pushing new samples and the end of dataset
'''
# handler_binance.print_to_file('out1.txt')
# # i = 0
# # while i != 3:
# #     time.sleep(60)
# #     handler_binance.get_recent_OHLC()
# #     i += 1
# #
# # handler_binance.print_to_file('out2.txt')

'''
Feed Forward neural network
'''

# for e in range(epochs):
#     for i in range(len(binance.dataset)):
#         t = binance.dataset.iloc[i]
#         tm1 = binance.dataset.iloc[i-1]
#         del t['Date']
#         del tm1['Date']
#         t = t.values
#         tm1 = tm1.values
#
#         inputs = np.concatenate([t, tm1])
#         inputs = inputs.reshape(-1, 1)
#         inputs = services.normalize(inputs)
#         inputs = inputs.reshape(28, )
#         inputs = inputs * 0.99 + 0.01
#         #
#         # print("Input value -> index {}\n value {}".format(i, inputs))
#         # print("Target value -> index {}\n value {}".format(i, targets[i]))
#
#         nn.train(inputs, targets[i])
#         print(i, e)


# k = binance.dataset.iloc[45]
# km = binance.dataset.iloc[44]
# del k['Date']
# del km['Date']
# k = k.values
# km = km.values
# inp = np.concatenate([k, km])
# inp = inp.reshape(-1, 1)
# inp = services.normalize(inp)
# inp = inp.reshape(28, )
# inp = inp * 0.99 + 0.01
#
# ans = nn.query(inp)
#
# print("\n")
# print(ans)



'''
LSTM neural network
'''
