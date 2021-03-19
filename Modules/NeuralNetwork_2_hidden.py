import numpy as np
from scipy import special
from Constants import *
from Modules.services import MyLogger


class NeuralNetwork:

    def __init__(self, input_nodes: int, hidden_nodes: int, hidden_nodes2: int, output_nodes: int,
                 learning_rate: float):

        # Initialize nodes
        self.inodes = input_nodes
        self.hnodes1 = hidden_nodes
        self.hnodes2 = hidden_nodes2
        self.onodes = output_nodes

        # Initialize learning rate
        self.lr = learning_rate

        # Initialize weights
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes1, self.inodes))
        self.whh = np.random.normal(0.0, pow(self.hnodes1, -0.5), (self.hnodes2, self.hnodes1))
        self.who = np.random.normal(0.0, pow(self.hnodes2, -0.5), (self.onodes, self.hnodes2))

        # Set activation function
        self.activation_function = lambda x: special.expit(x)

    def train(self, input_list: list, target_list: list):

        inputs = np.array(input_list, ndmin=2).T
        targets = np.array(target_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs  # [1 0]-[0 0]=[1, 0]   [0 0]-[1 1]=[-1, -1]

        # Variable penalising for different outputs
        if targets[0] == 1.0 and targets[1] == 0.01:
            output_errors = output_errors * 1.0
        elif targets[0] == 0.01 and targets[1] == 1.0:
            output_errors = output_errors * 1.0
        else:
            output_errors = output_errors * HOLD_ERROR_PENALIZING

        hidden_errors2 = np.dot(self.who.T, output_errors)
        hidden_errors1 = np.dot(self.whh.T, hidden_errors2)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs2.T)
        self.whh += self.lr * np.dot((hidden_errors2 * hidden_outputs2 * (1.0 - hidden_outputs2)), hidden_outputs1.T)
        self.wih += self.lr * np.dot((hidden_errors1 * hidden_outputs1 * (1.0 - hidden_outputs1)), inputs.T)

    def query(self, input_list: list) -> list:

        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def save_to_file(self):
        MyLogger.write_console("Saving neural network to file!\n")
        try:
            np.savetxt("Data/Neurons/l1.csv", self.wih, delimiter=",")
            np.savetxt("Data/Neurons/l2.csv", self.whh, delimiter=",")
            np.savetxt("Data/Neurons/l3.csv", self.who, delimiter=",")
        except Exception:
            raise ValueError("Something wrong occurred while saving neural network to file!")

    def load_from_file(self):
        MyLogger.write_console("Loading neural network from file!")
        try:
            self.wih = np.loadtxt("Data/Neurons/l1.csv", delimiter=",")
            self.whh = np.loadtxt("Data/Neurons/l2.csv", delimiter=",")
            self.who = np.loadtxt("Data/Neurons/l3.csv", delimiter=",")
            # input("Loading successfully, press enter for continue..")

        except Exception:
            raise ValueError("Cannot load neural network from file, file does not exists!")
