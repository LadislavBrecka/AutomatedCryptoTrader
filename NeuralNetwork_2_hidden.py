import numpy as np
from scipy import special


class NeuralNetwork:

    def __init__(self, inputnodes, hiddennodes, hiddennodes2, outputnodes, learningrate):

        # Initialize nodes
        self.inodes = inputnodes
        self.hnodes1 = hiddennodes
        self.hnodes2 = hiddennodes2
        self.onodes = outputnodes

        # Initialize learning rate
        self.lr = learningrate

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

        output_errors = targets - final_outputs
        hidden_errors2 = np.dot(self.who.T, output_errors)
        hidden_errors1 = np.dot(self.whh.T, hidden_errors2)

        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs2.T)
        self.whh += self.lr * np.dot((hidden_errors2 * hidden_outputs2 * (1 - hidden_outputs2)), hidden_outputs1.T)
        self.wih += self.lr * np.dot((hidden_errors1 * hidden_outputs1 * (1 - hidden_outputs1)), inputs.T)

    def query(self, input_list: list) -> list:

        inputs = np.array(input_list, ndmin=2).T

        hidden_inputs1 = np.dot(self.wih, inputs)
        hidden_outputs1 = self.activation_function(hidden_inputs1)

        hidden_inputs2 = np.dot(self.whh, hidden_outputs1)
        hidden_outputs2 = self.activation_function(hidden_inputs2)

        final_inputs = np.dot(self.who, hidden_outputs2)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs




