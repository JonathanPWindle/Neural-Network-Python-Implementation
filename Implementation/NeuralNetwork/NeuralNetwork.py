import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def f(x):
    return x*2


class NeuralNetwork:

    def __init__(self, inputNodes, hiddenNodes, outputNodes):
        # Number of input Nodes
        self.inputNodes = inputNodes
        # Number of hidden Nodes
        self.hiddenNodes = hiddenNodes
        # Number of output Nodes
        self.outputNodes = outputNodes

        # Weights effecting input -> hidden layers
        self.weightsInHid = np.zeros(shape=(self.hiddenNodes, self.inputNodes))
        # Weights effecting hidden -> output layers
        self.weightsHidOut = np.zeros(shape=(self.outputNodes, self.hiddenNodes))
        # Start with random weights
        self.weightsInHid = 2 * np.random.rand(self.hiddenNodes, self.inputNodes) -1
        self.weightsHidOut = 2 * np.random.rand(self.outputNodes, self.hiddenNodes) - 1

        # Bias associated with hidden nodes
        self.biasHid = np.ones(shape=(self.hiddenNodes, 1))
        self.biasHid = 2 * np.random.rand(self.hiddenNodes, 1) - 1
        # Bias associated with output nodes
        self.biasOut = np.ones(shape=(self.outputNodes, 1))
        self.biasOut = 2 * np.random.rand(self.outputNodes, 1) - 1

    def feedForward(self, input, activeFunc):

        # Generate hidden outputs
        # Multiply weights by the inputs
        hidden = np.matmul(self.weightsInHid,input)
        # Add the bias onto the calculated hidden outputs
        hidden = np.add(self.biasHid,hidden)
        # Apply the activation function to the hidden outputs
        activationFunc = np.vectorize(activeFunc)
        hidden = activationFunc(hidden)

        # Generate outputs
        # Multiply weights by hidden node output
        output = np.matmul(self.weightsHidOut, hidden)
        # Add the bias
        output = np.add(self.biasOut, output)
        # Apply the activation function
        output = activationFunc(output)

        return output

