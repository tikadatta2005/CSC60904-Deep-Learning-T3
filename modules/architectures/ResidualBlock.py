import torch

class ResidualBlock(torch.nn.Module):
    
    def __init__(self, *layers):
        super().__init__()
        self.layers = torch.nn.Sequential(*layers)
    
    def forward(self, x):
        return  x + self.layers(x)