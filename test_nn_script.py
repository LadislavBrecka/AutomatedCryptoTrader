from NN_Training import nn_train
from Modules.services import MyLogger
import os

files = 0

for _, dirnames, filenames in os.walk("D:/Laco/SKOLA/FEI STU/Bakalarka/AutomatedCryptoTrader/Data/Datasets/Testing/"):
    files += len(filenames)
MyLogger.write_console("{} files detected!".format(files))

for i in range(1, files+1):
    nn_train('Data/Datasets/Testing/{}.csv'.format(i), True, False, 1.0)