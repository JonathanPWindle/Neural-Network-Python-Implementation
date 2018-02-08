## Basic Neural Network

This is a small project to delve into the land of neural networks.

I'm primarily using tutorials from [The Coding Train](https://www.youtube.com/channel/UCvjgXvBlbQiydffZU7m1_aw) in his 
*Neural Networks - The Nature of Code* playlist to get started with this along with further research to support a greater
understanding.

---
---
##### Curent Capabilities:
* Can have any number of Nodes in each layer
    * Can only have one hidden layer at the moment
* Feed forward and train given JSON in the format:

```javascript
[
    {
        "inputs": [0,0],
        "target": 0
    },
    {
        "inputs": [0,1],
        "target": 1
    }
]
```

##### Future Ideas:
* Multiple hidden layers
* CNN capability
* Different activation functions