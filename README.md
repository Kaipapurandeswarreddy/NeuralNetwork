# Simple 2(inputs)→4(hidden layers)→1(ouput) PyTorch Model

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

```python
x = torch.tensor([[0.,0.],[0.,1.],[1.,0.],[1.,1,]])
y = torch.tensor([[0.],[1.],[1.],[0.]])
```
- Creates the input vectors 
- Inputs and target 

```python
loss_fn = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)
```
- Uses mean-squared error (MSE) as the loss function and the Adam optimizer with a learning rate of 0.01
```python
for _ in range(1000):
  opt.zero_grad()
  loss = loss_fn(model(x), y)
  loss.backward()
  opt.step()
```
- Standard training loop:
  - Zero optimizer gradients
  - Forward pass (model(x)), compute loss with `y`
  - Backpropagate
  - Update parameters

```python
print(model(x))
```
- Prints model outputs after training.
