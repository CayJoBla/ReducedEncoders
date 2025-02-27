# modeling_reduce.py

from torch import nn
from collections import OrderedDict
from transformers.activations import ACT2FN
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoConfig,
)

from .configuration_reduced import ReducedConfig


class DimReshape(nn.Module):
    """Layer that changes the dimensionality of the hidden space / embeddings.
    Currently used in both the DimReduce and DimReconstruct modules.
    Includes a linear layer, an activation function, and a dropout layer.
    """
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if isinstance(config.hidden_act, str):
            self.activation = ACT2FN[config.hidden_act]
        else:
            self.activation = config.hidden_act
        dropout_prob = 0.0
        if hasattr(config, 'hidden_dropout_prob'):
            dropout_prob = config.hidden_dropout_prob
        elif hasattr(config, 'attention_dropout'):
            dropout_prob = config.attention_dropout
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, x):
        output = self.linear(x)
        output = self.activation(output)
        output = self.dropout(output)
        return output


class DimReduce(nn.Sequential):
    """
    Module to insert between the base transformer model and the model head in 
    order to reduce the dimensionality of the hidden states.

    Args:
        config (PretrainedConfig): Configuration for the base model. 
            Should include the hidden_size and reduction_sizes parameters.
        modules (OrderedDict): An optional ordered dictionary of modules from
            which to load the reduction layers.
    """
    def __init__(self, config, modules=None):
        input_size = config.hidden_size
        self.reduction_sizes = config.reduction_sizes
        DimReduceLayer = DimReshape     # Default reshape layer
        
        if modules is None:
            modules = OrderedDict()
            for i, reduction_size in enumerate(self.reduction_sizes):   
                modules[str(i)] = DimReduceLayer(input_size, reduction_size, 
                                                    config)
                input_size = reduction_size
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)]
            )
    
        super().__init__(modules)


class DimExpand(nn.Sequential):
    """A module used during pretraining of a compressed model for reconstructing 
    the full-size hidden states.

    Args:
        config (PretrainedConfig): Configuration for the base model. 
            Should include the hidden_size and reduction_sizes parameters.
        modules (OrderedDict): An optional ordered dictionary of modules from 
            which to load the expansion layers. 
    """
    def __init__(self, config, modules=None):
        input_size = config.reduced_size
        self.expansion_sizes = config.reduction_sizes[-2::-1] 
        self.expansion_sizes.append(config.hidden_size)
        DimExpandLayer = DimReshape     # Default reshape layer
        
        if modules is None:
            modules = OrderedDict()
            for i, expansion_size in enumerate(self.expansion_sizes):   
                modules[str(i)] = DimExpandLayer(input_size, expansion_size, 
                                                    config)
                input_size = expansion_size
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict(
                [(str(idx), module) for idx, module in enumerate(modules)]
            )
    
        super().__init__(modules)


class DimReduceLoader(PreTrainedModel):
    """Used for loading only the dimensionality reduction module from a 
    pretrained model checkpoint. 
    """
    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.reduce = DimReduce(config)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.pop('config', 
                            AutoConfig.from_pretrained(*args, **kwargs))
        return super().from_pretrained(*args, config=config, **kwargs).reduce


class ReducedPreTrainedModel(PreTrainedModel):
    """An abstract class for defining common methods between reduced models."""
    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        config = self.config_class.from_config(config)
        super().__init__(config)

    def load_reduction(self, reduction_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained dimensionality reduction module
        into the reduced model.
        """
        self.reduce = DimReduceLoader.from_pretrained(
                            reduction_model_name_or_path, *args, **kwargs
                        )

    def _init_weights(self, module):
        """Default weight initialization."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, 
                                        std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, 
                                        std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        

    @staticmethod
    def _is_reduced_model(config):
        """Determine whether a model is a reduced model from the config."""
        return issubclass(config.__class__, ReducedConfig)

    @classmethod
    def from_pretrained(
        cls, 
        pretrained_model_name_or_path, 
        *model_args, 
        reduce_module=None, 
        base_model_class=None, 
        **kwargs
    ):
        # Load the config for the model (potentially a base model config)
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                            **kwargs)
        pretrained_is_reduced = cls._is_reduced_model(config)

        # Update config with provided parameters
        if "config" in kwargs: 
            config = kwargs.pop("config")

        # Load the model 
        if pretrained_is_reduced:
            model = super(ReducedPreTrainedModel, cls).from_pretrained(
                pretrained_model_name_or_path, 
                *model_args, 
                config=config, 
                **kwargs
            )
        else:
            # Load the base model
            if base_model_class is not None:
                base_model = base_model_class.from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    config=config, 
                    **kwargs
                )
            else:
                base_model = AutoModel.from_pretrained(
                    pretrained_model_name_or_path, 
                    *model_args, 
                    config=config, 
                    **kwargs
                )

            # Load the reduction module (if specified)
            if reduce_module is not None: 
                if type(reduce_module) is not DimReduce:
                    reduce_config = AutoConfig.from_pretrained(
                                        reduce_module, **kwargs)
                    reduce_module = DimReduceLoader.from_pretrained(
                                        reduce_module, config=reduce_config)

                # Override base model config from the structure of reduce_module
                # TODO: Consider modifying reduce_module to account for config
                config.reduction_sizes = reduce_module.reduction_sizes
                config.reduced_size = reduce_module.reduction_sizes[-1]

            # TODO: Currently, no warning is given when loading a base model 
            #       without a reduction. Should warn the user that reduction 
            #       weights are randomly initialized.

            # TODO: There is another issue where reduced model specific kwargs
            #       are passed to the base model when loading from pretrained.
            
            model = cls(
                config=config, 
                base_model=base_model, 
                reduce_module=reduce_module
            )
        
        return model