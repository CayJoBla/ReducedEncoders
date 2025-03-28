# modeling_reduce.py

from torch import nn
# from collections import OrderedDict
# from transformers import (
#     PreTrainedModel,
#     AutoModel,
#     AutoConfig,
# )

# TODO: Figure out how to make these compatible with both HF and Sentence Transformers
# TODO: Implement a ReducedEmbedder (name?) class that will load in the pretrained
#       reduction as well as the pretrained base model. Ideally, there will be a 
#       version for HF and Sentence Transformers, and the reduction checkpoint will
#       not have to save the base model weights, they would be loaded directly from
#       the base model checkpoint (but allow for base and reduction to be loaded from
#       the same checkpoint in the case of a fine-tuned model).

class Resize(nn.Sequential):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation_function = nn.SiLU(),
        dropout: float = 0.1,
    ):
        super().__init__(
            nn.Linear(in_features, out_features, bias=bias),
            activation_function,
            nn.Dropout(dropout),
        )
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.activation_function = activation_function.__class__.__name__
        self.dropout = dropout
        
    def get_config_dict(self):
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
            "activation_function": self.activation_function,
            "dropout": self.dropout
        }
        
    def __repr__(self):
        return f"Resize({self.get_config_dict()})"

class MultiResize(nn.Sequential):
    def __init__(self, 
        sizes, 
        bias: bool = True, 
        activation_function = nn.SiLU(), 
        dropout: float = 0.1
    ):
        super().__init__()
        for i in range(len(sizes) - 1):
            in_features = sizes[i]
            out_features = sizes[i+1]
            self.append(Resize(in_features, out_features, bias=bias, 
                                activation_function=activation_function, 
                                dropout=dropout))
        self.sizes = sizes

    def get_config_dict(self):
        return {
            "sizes": self.sizes
        }
    
    def __repr__(self):
        return f"MultiResize({self.get_config_dict()})"



# class ReducedPreTrainedModel(PreTrainedModel):
#     """An abstract class for defining common methods between reduced models."""
#     config_class = None
#     base_model_prefix = ""

#     def __init__(self, config):
#         config = self.config_class.from_config(config)
#         super().__init__(config)

#     def load_reduction(self, reduction_model_name_or_path, *args, **kwargs):
#         """Load the weights of a pretrained dimensionality reduction module
#         into the reduced model.
#         """
#         self.reduce = DimReduceLoader.from_pretrained(
#                             reduction_model_name_or_path, *args, **kwargs
#                         )

        

#     @staticmethod
#     def _is_reduced_model(config):
#         """Determine whether a model is a reduced model from the config."""
#         return issubclass(config.__class__, ReducedConfig)

#     @classmethod
#     def from_pretrained(
#         cls, 
#         pretrained_model_name_or_path, 
#         *model_args, 
#         reduce_module=None, 
#         base_model_class=None, 
#         **kwargs
#     ):
#         # Load the config for the model (potentially a base model config)
#         config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
#                                             **kwargs)
#         pretrained_is_reduced = cls._is_reduced_model(config)

#         # Update config with provided parameters
#         if "config" in kwargs: 
#             config = kwargs.pop("config")

#         # Load the model 
#         if pretrained_is_reduced:
#             model = super(ReducedPreTrainedModel, cls).from_pretrained(
#                 pretrained_model_name_or_path, 
#                 *model_args, 
#                 config=config, 
#                 **kwargs
#             )
#         else:
#             # Load the base model
#             if base_model_class is not None:
#                 base_model = base_model_class.from_pretrained(
#                     pretrained_model_name_or_path, 
#                     *model_args, 
#                     config=config, 
#                     **kwargs
#                 )
#             else:
#                 base_model = AutoModel.from_pretrained(
#                     pretrained_model_name_or_path, 
#                     *model_args, 
#                     config=config, 
#                     **kwargs
#                 )

#             # Load the reduction module (if specified)
#             if reduce_module is not None: 
#                 if type(reduce_module) is not DimReduce:
#                     reduce_config = AutoConfig.from_pretrained(
#                                         reduce_module, **kwargs)
#                     reduce_module = DimReduceLoader.from_pretrained(
#                                         reduce_module, config=reduce_config)

#                 # Override base model config from the structure of reduce_module
#                 # TODO: Consider modifying reduce_module to account for config
#                 config.reduction_sizes = reduce_module.reduction_sizes
#                 config.reduced_size = reduce_module.reduction_sizes[-1]

#             # TODO: Currently, no warning is given when loading a base model 
#             #       without a reduction. Should warn the user that reduction 
#             #       weights are randomly initialized.

#             # TODO: There is another issue where reduced model specific kwargs
#             #       are passed to the base model when loading from pretrained.
            
#             model = cls(
#                 config=config, 
#                 base_model=base_model, 
#                 reduce_module=reduce_module
#             )
        
#         return model