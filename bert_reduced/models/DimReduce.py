# DimReduce.py

from transformers import PreTrainedModel, AutoConfig
from torch import nn
from .DimReduceLayer import DimReduceLayer
from collections import OrderedDict

class DimReduce(nn.Sequential):
    """
    Module to insert between the base transformer model and the model head in order to reduce 
    the dimensionality of the hidden states. Uses the hidden_activation parameter, the 
    hidden_size parameter, and the custom reduction_sizes parameter from the base model 
    configuration.

    Args:
        config (PretrainedConfig): Configuration for the base model. Should include the hidden_size
            and reduction_sizes parameters for the dimensions of each layer of this module.
        modules (OrderedDict): An optional ordered dictionary of modules to load the reduction
            layers from. If not specified, the reduction layers will be randomly initialized, 
            using the sizes from the reduction_sizes parameter in the configuration.
    """
    def __init__(self, config, modules=None):
        input_size = config.hidden_size
        self.reduction_sizes = config.reduction_sizes
        
        if modules is None:
            modules = OrderedDict()
            for i, reduction_size in enumerate(self.reduction_sizes):   
                modules[str(i)] = DimReduceLayer(input_size, reduction_size, config)
                input_size = reduction_size
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
    
        super().__init__(modules)


class DimReduceLoader(PreTrainedModel):
    """A wrapper for the DimReduce module to load the pretrained reduction weights from a 
    Huggingface hub model. This module is not meant to be run or included in a model."""
    # These class values must remain unassigned so that the base_model_prefix is not used
    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.reduce = DimReduce(config)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.pop('config', AutoConfig.from_pretrained(*args, **kwargs))
        return super().from_pretrained(*args, config=config, **kwargs).reduce

        