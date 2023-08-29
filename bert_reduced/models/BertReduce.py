# BertReduce.py

from torch import nn
from .BertReductionLayer import BertReductionLayer

class BertReduce(nn.Module):
    """
    Module to insert between the base BERT model and the model head in order to reduce 
    the dimensionality of the hidden states. Uses the hidden_activation parameter from
    the base BERT configuation and the custom reduced_size parameter.

    Args:
        config (BertConfig): Configuration for the BERT model. Should also include the 
            reduced_size parameter for the final size of the output of this module.
        inter_sizes (tuple): A sequence of intermediate layer sizes to reduce the
            dimensionality over multiple linear layers.
    """
    def __init__(self, config, inter_sizes=()):
        super().__init__()
        input_size = config.hidden_size
        output_size = config.reduced_size
        
        self.layer = nn.ModuleList()
        for inter_size in inter_sizes:   
            self.layer.append(BertReductionLayer(input_size, inter_size, config))
            input_size = inter_size
        self.layer.append(BertReductionLayer(input_size, output_size, config))

    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states
        