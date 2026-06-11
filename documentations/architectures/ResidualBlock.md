# ResidualBlock

This is a simple class or bluprint that follows principle of ResNet:
``` math
H(x) = F(x) + x
```

## Structure of Class
The class has two functions:
* A Constructor
* Forward Module

### Constructor
The constructor takes multiple parameters where each should be a Neural Layer. These all Neural Layers is then forwaded to `torch.nn.Sequential`. This Sequentual module helps the provided layers to be stacked in a sequential order so it flows linearly from start to end.
``` text
[Conv1]->[ReLu]->[Conv2]->[ReLu]->[Conv3]->[ReLU]
```
This layer is then stored in `self.layers`

### Forward Module
The Forward Module is the Forward Propagation of the `ResidualBlock` class. Here the input x is added with layers stored in `self.layers`.

## Use Case 
The Block is very easy to utilize. A small demo for the utilization is given below:

``` 
Assumption 1: Torch is Already Loaded
Assumption 2: x is the input Tensor 
```

1. Initialize class using Modules
    ``` py
    residual = ResidualBlock(
        #Here goes all the Layers
        # First Layer
        torch.nn.Conv2D(...),
        torch.nn.ReLu(),
        torch.nn.MaxPool2D(...),

        # Second Layer
        torch.nn.Conv2D(...),
        torch.nn.ReLu(),
    )
    ```
2. Use Forward Module.
    ``` text
    The ResidualBlock class extends torch.nn.Module. This allows the forward to be directly called as residual() if residual = ResidualBlock(...).

    Therefore, we don't use residual.forward()
    ```
    Usecase:
    ``` py
    residual(x) 
    ```
From above mentioned code the layer works in the following top to bottom way:
```
            x (Input)
                 |
           [First Layer]
                 |
           [Second Layer]
                 |
            (output + x)
```
## References
* https://www.comet.com/site/blog/resnet-how-one-paper-changed-deep-learning-forever/