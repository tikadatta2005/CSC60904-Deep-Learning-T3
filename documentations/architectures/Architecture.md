# Architecture
It's a simple blueprint for a model that dynamically accepts multiple Layers or blocks. This extends `torch.nn.Module`

## Structure of Architecture
The Architecture has three functions:
* Constructor
* Add Module
* Forward Module

### Constructor
The Constructure here is basically used to:
* Initialize `torch.nn.Module()` using `super().__init__()`
* Creating `ModuleList()` to hold list of Modules or Layers

### Add Module
This is the module used to add new `torch.nn Modules` or `Layers`. This sequentially adds each modules from parameters to `torch.nn.Sequential()`

### Forward Module
This is the Feed Forward Module. This simply sends `x` to next module step by step. Here the output of `L1` is the input for `L2`

## Usecase:
A simple usecase for the module is as follows:
### Notaions
``` text
x : Input Tensor
y : Label Tensor
L : Layer
Li : Layer no i. i is Variable.
criterion : Loss Function
optimizer : Any Optimizer like SGD and Adam
```
### Python Code
``` py
model = Architecture()
# shift model to cuda
model = model.to("cuda")

# Add Layers / Blocks
model.add(
    L1,
    L2,
    L3
)

# Feed Forward
y_pred = model(x)
```

Then the class can later utilized like:
``` py
# calculate loss
loss = criterion(y_pred, y)

# backward propagaition
loss.backward()
# update gradients
optimizer.step()
```