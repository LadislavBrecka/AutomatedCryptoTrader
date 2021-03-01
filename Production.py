import time
import DataHandling

'''
Pushing new samples and the end of dataset
'''
print("Printing from main.py: Start adding new samples")
i = 0
while i != 3:
    time.sleep(60)
    DataHandling.binance.get_recent_OHLC()
    i += 1
