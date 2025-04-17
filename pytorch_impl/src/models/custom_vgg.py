import torch
from torch import nn

class CustomVgg(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int)-> None:
        super().__init__()

        self.block_1 = nn.Sequential(nn.Conv2d(in_channels=input_shape,
                                               out_channels=hidden_units,
                                               kernel_size=3,
                                               stride=1,
                                               padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(in_channels=hidden_units,
                                               out_channels=hidden_units,
                                               kernel_size=3,
                                               stride=1,
                                               padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
    
        self.block_2 = nn.Sequential(nn.Conv2d(hidden_units,hidden_units,kernel_size=3,stride=1,padding=0),
                                    nn.ReLU(),
                                    nn.Conv2d(hidden_units,hidden_units,kernel_size=3,stride=1,padding=0),
                                    nn.ReLU(),
                                    nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Linear(in_features = hidden_units*13*13, out_features = output_shape))
        

    def forward(self, input):
        x = self.block_1(input)
        x = self.block_2(x)
        return self.classifier(x)    