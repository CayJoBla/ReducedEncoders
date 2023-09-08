# BertReduce.py

from torch import nn
from .BertReductionLayer import BertReductionLayer
from collections import OrderedDict

class BertReduce(nn.Sequential):
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
    def __init__(self, config, inter_sizes=(), modules=None):
        input_size = config.hidden_size
        output_size = config.reduced_size
        
        if modules is None:
            modules = OrderedDict()
            for i, inter_size in enumerate(inter_sizes):   
                modules[str(i)] = BertReductionLayer(input_size, inter_size, config)
                input_size = inter_size
            modules[str(i+1)] = BertReductionLayer(input_size, output_size, config)
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
    
        super().__init__(modules)

        