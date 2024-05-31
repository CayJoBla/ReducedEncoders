# modeling_reduce.py

from torch import nn
from collections import OrderedDict
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel
from transformers.modeling_outputs import BaseModelOutputWithNoAttention
import warnings

from .modeling_utils import get_activation


class DimReshape(nn.Module):
    """Layer that changes the dimensionality of the hidden space / embeddings.
    Currently used in both the DimReduce and DimExpand modules.
    Note that the LayerNorm, activation function, and Dropout layers are 
    applied before the dense layer.

    Parameters:
        input_size (int): Size of the hidden state inputs.
        output_size (int): Size of the hidden state outputs.
        config (ReducedConfig): Reduced model configuration.
    """
    def __init__(self, input_size, output_size, config):
        super().__init__()
        # self.layernorm = nn.LayerNorm(input_size, eps=config.layer_norm_eps)
        self.activation = get_activation(config.hidden_act)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.dense = nn.Linear(input_size, output_size)
        
    def forward(self, x):
        # x = self.layernorm(x)
        output = self.activation(x)
        output = self.dropout(output)
        embedding = self.dense(output)
        return embedding      


class DimReduce(nn.Sequential):
    """Module to insert after the base encoder model in order to reduce the dimensionality 
    of the hidden states. Utilizes the 'hidden_size', 'hidden_act', 'hidden_dropout_prob', 
    'layer_norm_eps', 'reduction_sizes', and 'reduced_size' parameters from the model 
    configuration to determine the structure of the reduction layers.

    Parameters:
        config (ReducedConfig): Reduced model configuration with the above parameters.
    """
    def __init__(self, config):     # TODO: Should I worry about how the weights are initialized here
        input_size = config.hidden_size
        reduction_sizes = config.reduction_sizes

        reduction_layers = []
        for reduction_size in reduction_sizes:
            reduction_layers.append(DimReshape(input_size, reduction_size, config))
            input_size = reduction_size
        super().__init__(*reduction_layers)
    
    def forward(self, x): 
        embeddings = (x,)
        for layer in self:
            embeddings += (layer(embeddings[-1]),)
        return embeddings[1:]


class DimExpand(nn.Sequential):
    """Module to insert after the reduction module in order to reconstruct the full size 
    hidden states from the reduced hidden states. Generally used for pretraining the 
    reduction module with a reconstruction loss.
    Utilizes the 'hidden_size', 'hidden_act', 'hidden_dropout_prob', 'layer_norm_eps', 
    'reduction_sizes', and 'reduced_size' parameters from the model configuration to 
    determine the structure of the reduction layers.

    The structure of the model closely follows that of the DimReduce module, but reverses
    the order of the reduction_sizes parameter for the intermediate layer sizes to go from
    smallest to largest instead of largest to smallest.

    Parameters:
        config (ReducedConfig): Reduced model configuration with the above parameters.
    """
    def __init__(self, config):
        input_size = config.reduced_size
        expansion_sizes = config.reduction_sizes[-2::-1] + [config.hidden_size]

        expansion_layers = []
        for expansion_size in expansion_sizes:
            expansion_layers.append(DimReshape(input_size, expansion_size, config))
            input_size = expansion_size
        super().__init__(*expansion_layers)
    
    def forward(self, x): 
        embeddings = (x,)
        for layer in self:
            embeddings += (layer(embeddings[-1]),)
        return embeddings[1:]
    

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


class ReducedPreTrainedModel(PreTrainedModel):
    """An abstract class for defining common methods between reduced models."""
    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        config = self.config_class.from_config(config)
        super().__init__(config)

    def load_reduction(self, reduction_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained dimensionality reduction module into the reduced model."""
        self.reduce = DimReduceLoader.from_pretrained(reduction_model_name_or_path, *args, **kwargs)

    @staticmethod
    def _is_reduced_model(config):
        """Determine whether a model is a reduced model from the config."""
        # TODO: May need an update for resiliency to changes in config / other configs with those parameters
        return "reduction_sizes" in config.__dict__ or "reduced_size" in config.__dict__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, reduce_module=None, 
                        base_model_class=None, **kwargs):
        # Load the config for the model (potentially a base model config)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_reduced_model = cls._is_reduced_model(config)

        # Update config with provided parameters
        if "config" in kwargs: 
            config.__dict__.update(kwargs.pop("config").__dict__)

        # Load the model (different depending on whether this is a reduced model or not)
        if is_reduced_model:
            model = super(ReducedPreTrainedModel, cls).from_pretrained(pretrained_model_name_or_path, 
                                                                       *model_args, config=config, **kwargs)
        else:
            # Load the base model
            if base_model_class is not None:
                base_model = base_model_class.from_pretrained(pretrained_model_name_or_path, 
                                                                *model_args, config=config, **kwargs)
            else:
                base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, 
                                                        *model_args, config=config, **kwargs)

            # Load the reduction module (if specified)
            if reduce_module is not None: 
                if type(reduce_module) is not DimReduce:
                    reduce_config = AutoConfig.from_pretrained(reduce_module, **kwargs)
                    reduce_module = DimReduceLoader.from_pretrained(reduce_module, config=reduce_config)

                # Currently, overrides the base model reduce configuration params from the structure of reduce_module
                config.reduction_sizes = reduce_module.reduction_sizes
                config.reduced_size = reduce_module.reduction_sizes[-1]

                model = cls(config=config, base_model=base_model, reduce_module=reduce_module)
            else:
                warnings.warn(f"The {cls.__name__} model is intended to have a dimensionality reduction "
                                "module, but was loaded from a pretrained model without reduction. Loading "
                                "the model with a randomly intialized reduction. To change this, either "
                                "specify the `reduction_model_name_or_path` argument when loading from a "
                                "pretrained model, or load a reduction separately using the `load_reduction()` "
                                "method.")
                model = cls(config=config, base_model=base_model, reduce_module=reduce_module)
        
        return model