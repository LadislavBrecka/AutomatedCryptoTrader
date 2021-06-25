from Modules.NN_Training import nn_train
from Modules.services import MyLogger
import os


'''
Console application
'''
files = 0

for _, dirnames, filenames in os.walk("D:/Laco/SKOLA/FEI STU/Bakalarka/AutomatedCryptoTrader/Data/Datasets/Testing/"):
    files += len(filenames)
MyLogger.write_console("{} files detected!".format(files))

vec_conf_mat = []
for i in range(1, files+1):
    percentage_diff = nn_train('Data/Datasets/Testing/{}.csv'.format(i), True, False, 1.0)
    vec_conf_mat.append(percentage_diff)

conf_mat = open("Outputs/conf_matrix.txt", 'w')
for r in vec_conf_mat:
    conf_mat.write("+ " + str(r) + " %" + "\n")
conf_mat.close()