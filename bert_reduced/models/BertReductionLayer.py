# BertReductionLayer.py

from torch import nn
from transformers.activations import ACT2FN

class BertReductionLayer(nn.Module):
    """
    Layer of the BertReduce module. Includes a linear layer, an activation function, 
    and a dropout layer.
    """
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        if isinstance(config.hidden_act, str):
            self.reduce_act_fn = ACT2FN[config.hidden_act]
        else:
            self.reduce_act_fn = config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        output = self.dense(x)
        output = self.reduce_act_fn(output)
        output = self.dropout(output)
        return output
