# configuration_mpnet_reduced.py

from transformers import MPNetConfig, AutoConfig

class MPNetReducedConfig(MPNetConfig):
    """This class wraps the MPNetConfig class to add the dimensionality reduction parameters and the pooling mode"""

    model_type = "mpnet_reduced"

    def __init__(self, *args, reduction_sizes=(48,), pooling_mode="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction_sizes = reduction_sizes
        self.reduced_size = reduction_sizes[-1]
        self.pooling_mode = pooling_mode

    @classmethod
    def from_config(cls, config):
        if isinstance(config, MPNetConfig):
            config = cls(**config.__dict__)
        else:
            raise ValueError("Parameter config should be an instance of class `MPNetConfig`.")
        config.model_type = cls.model_type
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(config)