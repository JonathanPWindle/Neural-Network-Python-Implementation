import numpy as np
import NeuralNetwork.NeuralNetwork as nn
import json
import random

myNN = nn.NeuralNetwork(2, 5, 1)
inputs = np.random.rand(2, 1)
targets = 1



# print(myNN.feedForward(np.asmatrix([[0],[0]]),nn.sigmoid))
#print(myNN.feedForward(inputs, nn.sigmoid))

inputs = """
[
    { "inputs": [0,0],
      "targets": 0
    },
    {
      "inputs": [0,1],
      "targets": 1
    },
    {
      "inputs": [1,0],
      "targets": 1
    },
    {
      "inputs": [1,1],
      "targets": 0
    }
]
"""

inputs = json.loads(inputs)
print(inputs)
print(random.choice(inputs)['inputs'])

counts = [0,0,0,0]
for i in range(0,100000):
    data = random.choice(inputs)
    myNN.train(np.asmatrix(data['inputs']).transpose(), data['targets'], 0.1)


print(myNN.feedForward(np.asmatrix([0, 0]).transpose(), nn.sigmoid))
print(myNN.feedForward(np.asmatrix([0, 1]).transpose(), nn.sigmoid))
print(myNN.feedForward(np.asmatrix([1, 0]).transpose(), nn.sigmoid))
print(myNN.feedForward(np.asmatrix([0, 0]).transpose(), nn.sigmoid))
