# ReducedPreTrainedModel.py

from transformers import PreTrainedModel, AutoConfig, PretrainedConfig, AutoModel
import warnings

from .DimReduce import DimReduce, DimReduceLoader

class ReducedPreTrainedModel(PreTrainedModel):
    """An abstract class for defining common methods between reduced models."""
    config_class = None
    base_model_prefix = ""
    
    def _initialize_config(self, config: PretrainedConfig, reduction_sizes=(48,)):
        """Ensure that the model configuration contains dimensionality reduction parameters, 
        setting default values if they are not specified. Assign the config to the model.

        Parameters:
            config (PretrainedConfig): The configuration object for the model. These
                parameters are prioritized over the defaults
            reduction_sizes (tuple): A sequence of reduction layer sizes. This is meant
                to be a default set by the model. Default is one layer with dimension 48.
        """
        # Prioritize the values in the config
        reduction_sizes = config.__dict__.get("reduction_sizes", reduction_sizes)
            
        # Assign the values to the config
        config.reduction_sizes = reduction_sizes
        config.reduced_size = reduction_sizes[-1]
        self.config = config

    def load_reduction(self, reduction_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained dimensionality reduction module into the reduced model."""
        self.reduce = DimReduceLoader.from_pretrained(reduction_model_name_or_path, *args, **kwargs)

    @staticmethod
    def _is_reduced_model(config):
        """Determine whether a model is a reduced model from the config."""
        return "reduction_sizes" in config.__dict__ or "reduced_size" in config.__dict__

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, reduce_module=None, 
                        base_model_class=None, **kwargs):
        # Load the config for the model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_reduced_model = cls._is_reduced_model(config)

        # Update config with provided parameters
        if "config" in kwargs: 
            config.__dict__ = config.__dict__.update(kwargs["config"].__dict__)

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
            