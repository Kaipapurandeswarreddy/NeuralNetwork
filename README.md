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

- `nn.Linear(2, 4)` — affine transform from a 2-dimensional input to 4 hidden units.
- `nn.ReLU()` — element-wise ReLU activation introduces non-linearity.
- `nn.Linear(4, 1)` — affine transform from 4 hidden units to 1 output value.
## Tensor shapes

- Input tensor should have shape `(batch_size, 2)`.
- Output tensor will have shape `(batch_size, 1)`.

Example:
```python
x = torch.randn(10, 2)   # batch of 10 samples
y = model(x)             # y.shape -> torch.Size([10, 1])
```

## Minimal training step example

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

criterion = nn.MSELoss()                 # for regression
optimizer = optim.SGD(model.parameters(), lr=1e-2)

# dummy data
inputs = torch.randn(32, 2)              # batch of 32
targets = torch.randn(32, 1)             # regression targets

model.train()
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
```

For binary classification, replace the loss with `nn.BCEWithLogitsLoss()` and provide targets in shape `(batch_size, 1)` with values 0 or 1.

## Tips & notes

- Move the model and data to the same device: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` then `model.to(device)` and `inputs = inputs.to(device)`.
- Choose appropriate dtype (e.g., `torch.float32`) for inputs and model parameters.
- For reproducibility, set random seeds: `torch.manual_seed(seed)`.
- You can add weight initialization if desired:
  ```python
  for m in model.modules():
      if isinstance(m, nn.Linear):
          nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
          nn.init.zeros_(m.bias)
  ```
- If you expect a single scalar per sample (not a 1-dim tensor), you can squeeze the output with `outputs = outputs.squeeze(-1)` to get shape `(batch_size,)`.

## Common errors

- AttributeError: module 'torch.nn' has no attribute 'sequential' — use `nn.Sequential` (capital `S`).
- Shape mismatch — ensure your input has shape `(N, 2)` where `N` is the batch size.

## License

Choose a license for your project (e.g., MIT) and add a `LICENSE` file if you plan to share the code.
