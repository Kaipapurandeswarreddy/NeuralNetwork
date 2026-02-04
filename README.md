# Simple 2→4→1 PyTorch Model

This repository contains a minimal feedforward neural network implemented in PyTorch. The network maps 2-dimensional inputs to a single scalar output using one hidden layer with ReLU activation.

## Model definition
```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 4),   # input dim = 2, hidden dim = 4
    nn.ReLU(),         # activation
    nn.Linear(4, 1),   # hidden dim = 4, output dim = 1
)
```
## What the model does

- `nn.Linear(2, 4)` — The input layer contains Two neurons and outputs to 4 Neurons
- `nn.ReLU()` — Its the activation between neurons
- `nn.Linear(4, 1)` — It takes the 4 neurons output as input and outputs as 1

