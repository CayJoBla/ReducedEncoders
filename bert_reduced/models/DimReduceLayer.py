# DimReduceLayer.py

from torch import nn
from transformers.activations import ACT2FN

class DimReduceLayer(nn.Module):
    """Layer of the DimReduce module, reducing the dimensionality of the hidden space. 
    Includes a linear layer, an activation function, and a dropout layer.
    """
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        output = self.linear(x)
        output = self.activation(output)
        output = self.dropout(output)
        return output
