from Modules.services import MyLogger
from Modules.NN_Training import nn_train
import os


'''
Console application
'''
files = 0

for _, dirnames, filenames in os.walk("D:/Laco/SKOLA/FEI STU/Bakalarka/AutomatedCryptoTrader/Data/Datasets/Active/"):
    files += len(filenames)
MyLogger.write_console("{} files detected!".format(files))
val_i = 0

try:
    validation_file_name = 'Data/Datasets/Active/validation.csv'
    validation_file = open(validation_file_name)
    val_i = 1
except IOError:
    validation_file_name = None
    val_i = 0

file_name = 'Data/Datasets/Active/1.csv'
nn_train(file_name, False, True, 0.0, validation_dataset_file_name=validation_file_name)


for i in range(2, files-1-val_i):
    file_name = "Data/Datasets/Active/{}.csv".format(str(i))
    nn_train(file_name, True, True, 0.0, validation_dataset_file_name=validation_file_name)

file_name = "Data/Datasets/Active/{}.csv".format(str(files-val_i))
percentage_diff = nn_train(file_name, True, False, 1.0, validation_dataset_file_name=validation_file_name)

conf_mat = open("Outputs/conf_matrix.txt")
conf_mat.write(percentage_diff)
conf_mat.close()
