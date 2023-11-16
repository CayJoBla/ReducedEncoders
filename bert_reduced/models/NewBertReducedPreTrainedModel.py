# BertReducedPreTrainedModel.py

from .ReducedPreTrainedModel import ReducedPreTrainedModel
from transformers import BertConfig

class BertReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced BERT models."""
    config_class = BertConfig
    base_model_prefix = "bert"

    def _initialize_config(self, config=None, reduction_sizes=(48,)):
        """Set the default configuration for reduced BERT models"""
        if config is None: config = BertConfig()
        super()._initialize_config(config, reduction_sizes=reduction_sizes)