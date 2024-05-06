# configuration_mpnet_reduced.py

from transformers import AutoConfig, PretrainedConfig

class ReducedConfig(PretrainedConfig):
    """A mixin class for defining common parameters between reduced model configurations."""

    def __init__(self, *args, reduction_sizes=[512,256,128,68,48], **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction_sizes = reduction_sizes
        self.reduced_size = reduction_sizes[-1]

    @classmethod
    def from_config(cls, config, **kwargs):
        """Create a reduced configuration from an existing configuration."""
        try:    # Get the base model configuration class
            base_model_config_class = cls.__bases__[1]    # ReducedConfig, [BaseModelConfig], ...
            if not issubclass(base_model_config_class, PretrainedConfig):
                raise ValueError(f"The second base class of the {cls.__name__} class should be a PretrainedConfig class.")
        except IndexError:
            raise ValueError(f"The ReducedConfig class should be used as a mixin class alongside a base model "
                             f"configuration class, but only got the {cls.__bases__} bases for the {cls.__name__} class.")

        # Initialize a new reduced configuration from the existing one
        if isinstance(config, base_model_config_class):
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
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(config)