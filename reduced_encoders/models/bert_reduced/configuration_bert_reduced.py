# configuration_bert_reduced.py

from transformers import BertConfig, AutoConfig

class BertReducedConfig(BertConfig):
    """This class wraps the BertConfig class to add the dimensionality reduction parameters."""

    model_type = "bert_reduced"

    def __init__(self, *args, reduction_sizes=(48,), **kwargs):
        super().__init__(*args, **kwargs)
        self.reduction_sizes = reduction_sizes
        self.reduced_size = reduction_sizes[-1]

    @classmethod
    def from_config(cls, config, **kwargs):
        if isinstance(config, BertConfig):
            config = cls(**config.__dict__, **kwargs)
        else:
            raise ValueError("Parameter config should be an instance of class `BertConfig`.")
        config.model_type = cls.model_type
        return config

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls.from_config(config)