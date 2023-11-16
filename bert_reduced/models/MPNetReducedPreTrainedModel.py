# MPNetReducedPreTrainedModel.py

from .ReducedPreTrainedModel import ReducedPreTrainedModel
from transformers import MPNetConfig

class MPNetReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced MPNet models."""
    config_class = MPNetConfig
    base_model_prefix = "mpnet"

    def _initialize_config(self, config=None, reduction_sizes=(48,)):
        """Set the default configuration for reduced MPNet models"""
        if config is None: config = MPNetConfig()
        super()._initialize_config(config, reduction_sizes=reduction_sizes)