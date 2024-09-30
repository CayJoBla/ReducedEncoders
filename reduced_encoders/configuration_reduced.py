# configuration_mpnet_reduced.py

from transformers import AutoConfig, PretrainedConfig
import warnings

class ReducedConfig(PretrainedConfig):
    """A mixin class for defining common parameters between reduced model configurations."""

    def __init__(
        self, 
        *args, 
        reduction_sizes=[512,256,128,68,48], 
        can_reduce_sequence=False,
        can_reduce_pooled=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.reduction_sizes = reduction_sizes
        self.reduced_size = reduction_sizes[-1]

        # Models for reduction pretraining should specify these parameters
        self.can_reduce_sequence = can_reduce_sequence
        self.can_reduce_pooled = can_reduce_pooled
        if not self.can_reduce_sequence and not self.can_reduce_pooled:
            warnings.warn("This config has both 'can_reduce_sequence' and "
                            "'can_reduce_pooled' set to False. Models using "
                            "this config will not apply reduction on forward "
                            "pass")

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create a reduced configuration from an existing configuration."""
        try:    # Get the base model configuration class
            # TODO: This is sloppy, find a better way to get the base model config class
            base_model_config_class = cls.__bases__[1]    # ReducedConfig, [BaseModelConfig], ...
            if not issubclass(base_model_config_class, PretrainedConfig):
                raise ValueError(f"The second base class of the {cls.__name__} class should be a PretrainedConfig class.")
        except IndexError:
            raise ValueError(f"The ReducedConfig class should be used as a mixin class alongside a base model "
                             f"configuration class, but only got the {cls.__bases__} bases for the {cls.__name__} class.")

        # Initialize a new reduced configuration from the existing one
        if isinstance(config, base_model_config_class):     # TODO: This doesn't allow BERT config to be used in MPNet model
            config = cls(**config.__dict__, **kwargs)
        else:
            raise ValueError(f"Parameter config should be an instance of class {base_model_config_class}.")

        try:    # Ensure the model type is set correctly
            config.model_type = cls.model_type
        except AttributeError:
            raise ValueError(f"The {cls.__name__} class should have a unique `model_type` attribute.")
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)    # TODO: This could cause issues when the path config does not match the base model config
        return cls.from_config(config)