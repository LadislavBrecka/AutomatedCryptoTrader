from Modules.BinanceOhlcHandler import BinanceOhlcHandler
from Constants import *
import time

h = BinanceOhlcHandler(BINANCE_PAIR)
h.get_dataset(12, interval='5m')

h.print_to_file('out1.txt')
i = 0
while i != 3:
    time.sleep(5*60)
    h.get_recent_OHLC()
    i += 1

h.print_to_file('out2.txt')

