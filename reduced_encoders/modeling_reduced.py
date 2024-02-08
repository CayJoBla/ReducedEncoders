# modeling_reduce.py

from torch import nn
from collections import OrderedDict
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel
from transformers.activations import ACT2FN
import warnings

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


class Decoder(nn.Sequential):
    """A module used during pretraining of a compressed model that transforms the 
    reduced model embeddings back into full-size model embeddings with the goal
    of matching the original embeddings as closely as possible.

    The structure of the model closely follows that of the DimReduce module, but reverses
    the order of the reduction_sizes parameter for the intermediate layer sizes to go from
    smallest to largest instead of largest to smallest.
    """
    def __init__(self, config, modules=None):
        input_size = config.reduced_size
        self.decoding_sizes = config.reduction_sizes[-2::-1] + [config.hidden_size]

        DecoderLayer = DimReduceLayer   # Use the same structure as the DimReduce module
        
        if modules is None:
            modules = OrderedDict()
            for i, decoding_size in enumerate(self.decoding_sizes):   
                modules[str(i)] = DecoderLayer(input_size, decoding_size, config)
                input_size = decoding_size
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
    
        super().__init__(modules)


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
            





        