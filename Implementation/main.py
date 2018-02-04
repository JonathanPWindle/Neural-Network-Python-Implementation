import numpy as np
import NeuralNetwork.NeuralNetwork as nn

myNN = nn.NeuralNetwork(3,2,1)
inputs = np.random.rand(3,1)
print(myNN.feedForward(inputs, nn.sigmoid))
