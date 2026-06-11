import torch

class Architecture(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        
    def add(self, *modules):
        self.layers.append(torch.nn.Sequential(*modules))
        return self
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x