from Modules.services import MyLogger
from NN_Training import nn_train
import os

files = 0

for _, dirnames, filenames in os.walk("D:/Laco/SKOLA/FEI STU/Bakalarka/AutomatedCryptoTrader/Data/Datasets/"):
    files += len(filenames)
MyLogger.write_console("{} files detected, scrapping first and last!".format(files))


file_name = 'Data/Datasets/1.csv'
nn_train(file_name, False, True, 0.0)


for i in range(2, files-2):
    file_name = "Data/Datasets/{}.csv".format(str(i))
    nn_train(file_name, True, True, 0.0)

file_name = "Data/Datasets/{}.csv".format(str(files-2))
nn_train(file_name, True, False, 1.0)
