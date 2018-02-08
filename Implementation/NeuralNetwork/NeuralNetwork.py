import numpy as np


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def dsigmoid(y):
    return y * (1-y)


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

    def feedForward(self, inputs, activeFunc):

        # Generate hidden outputs
        # Multiply weights by the inputs
        hidden = np.matmul(self.weightsInHid, inputs)
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

    def train(self,inputs, targets, learningRate):
        # Feed the inputs into the network
        outputs = self.feedForward(inputs, sigmoid)

        # Calculate the error
        outputError = np.subtract(targets, outputs)

        derivSigmoid = np.vectorize(dsigmoid)

        # Calculate gradient
        gradient = derivSigmoid(outputs)
        gradient = np.multiply(gradient,outputError)
        gradient = np.multiply(gradient, learningRate)

        # Calculate deltas
        hidden = np.matmul(self.weightsInHid, inputs)
        hidden = np.add(self.biasHid,hidden)
        activationFunc = np.vectorize(sigmoid)
        hidden = activationFunc(hidden)
        transposedHidden = hidden.transpose()
        hidOutDeltas = np.matmul(gradient, transposedHidden)

        # Adjust weights by deltas
        self.weightsHidOut = np.add(self.weightsHidOut, hidOutDeltas)
        # Adjust bias by its deltas
        self.biasOut = np.add(self.biasOut, gradient)


        # Calculate hidden output error
        transposedHidOutWeights = self.weightsHidOut.transpose()
        hiddenErrors = np.matmul(transposedHidOutWeights, outputError)

        # Calculate hidden gradient
        hiddenGradient = derivSigmoid(hidden)
        hiddenGradient = np.multiply(hiddenGradient,hiddenErrors)
        hiddenGradient = np.multiply(hiddenGradient,learningRate)

        # Calculate input->hidden deltas
        transposedInputs = inputs.transpose()
        inHidDeltas = np.matmul(hiddenGradient, transposedInputs)

        self.weightsInHid = np.add(self.weightsInHid, inHidDeltas)
        self.biasHid = np.add(self.biasHid, hiddenGradient)