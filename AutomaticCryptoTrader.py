from NN_Training import nn_train


for i in range(1, 12):
    nn_train('Data/Datasets/Archive/08.03-18.03_ETHEUR/{}.csv'.format(i), True, False, 1.0)
