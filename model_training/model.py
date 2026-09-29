import torch
import torch.nn as nn


class DigitClassifier(nn.Module):
    def __init__(self, input_layer, hidden_layer, output_layer):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_layer, hidden_layer),
            nn.ReLU(),
            nn.Linear(hidden_layer, output_layer)
        )


    def forward(self, x):
        x = torch.flatten(x, 1, -1)
        return self.network(x)