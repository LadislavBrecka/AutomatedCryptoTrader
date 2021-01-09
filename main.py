from BinanceOhlcHandler import BinanceOhlcHandler
from services import Services
from CoinMarketCapHandler import CoinMarketCapHandler
import Teacher
import numpy as np



'''
Setting and getting dataset
'''
import NeuralNetwork as nn_1_hidden
import NeuralNetwork_2_hidden as nn_2_hidden
nn = nn_2_hidden.NeuralNetwork(24, 10, 50, 1, 0.3)
# nn = nn_1_hidden.NeuralNetwork(24, 50, 1, 0.3)
services = Services()

binance_pair = 'BTCUSDT'
coin_name = 'bitcoin'

binance = BinanceOhlcHandler(binance_pair)
binance.get_dataset()
coinmarket = CoinMarketCapHandler(coin_name)
c = coinmarket.get_actual()
print(c)

look_back = 2
look_future = 5
epochs = 10

'''
Finding and setting buy/sell signals
'''
x = Teacher.generate_buy_sell_signal(binance.dataset)

binance.dataset['Buy'] = x[0]
binance.dataset['Sell'] = x[1]

'''
Plotting and printing
'''

binance.print_to_file('out1.txt')
binance.plot_candlestick(indicators=True)

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


targets = Teacher.get_target(look_back, look_future, binance.dataset)
print(len(targets))

for e in range(epochs):
    j = 0
    for i in range(1, len(binance.dataset)-look_future, look_back):
        t = binance.dataset.iloc[i]
        tm1 = binance.dataset.iloc[i-1]
        del t['Date']
        del tm1['Date']
        t = t.values
        tm1 = tm1.values

        inputs = np.concatenate([t, tm1])
        inputs = inputs.reshape(-1, 1)
        inputs = services.normalize(inputs)
        inputs = inputs.reshape(24, )
        inputs = inputs * 0.99 + 0.01
        #
        # print("Input value -> index {}\n value {}".format(i, inputs))
        # print("Target value -> index {}\n value {}".format(j, targets[j]))

        nn.train(inputs, targets[j])
        j = j + 1
        print(i, e)


k = binance.dataset.iloc[45]
km = binance.dataset.iloc[44]
del k['Date']
del km['Date']
k = k.values
km = km.values
inp = np.concatenate([k, km])
inp = inp.reshape(-1, 1)
inp = services.normalize(inp)
inp = inp.reshape(24, )
inp = inp * 0.99 + 0.01


ans = nn.query(inp)

print("\n")
print(ans)
print("BUY" if ans > 0.5 else "SELL")





'''
LSTM neural network
'''
