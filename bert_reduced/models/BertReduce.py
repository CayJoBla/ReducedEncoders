# BertReduce.py

from transformers import BertPreTrainedModel
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
            reduction_sizes parameter for the sizes of each reduction layer of this module.
        inter_sizes (tuple): A sequence of intermediate layer sizes to reduce the
            dimensionality over multiple linear layers.
    """
    def __init__(self, config, modules=None):
        input_size = config.hidden_size
        reduction_sizes = config.reduction_sizes
        
        if modules is None:
            modules = OrderedDict()
            for i, reduction_size in enumerate(reduction_sizes):   
                modules[str(i)] = BertReductionLayer(input_size, reduction_size, config)
                input_size = reduction_size
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
    
        super().__init__(modules)


class BertReduceLoader(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.reduce = BertReduce(config, modules=None)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return super().from_pretrained(*args, **kwargs)

        