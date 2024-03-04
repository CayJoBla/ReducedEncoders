# configuration_mpnet_reduced.py

from transformers import MPNetConfig, AutoConfig
from ...configuration_reduced import ReducedConfig

class MPNetReducedConfig(ReducedConfig, MPNetConfig):
    """This class wraps the MPNetConfig class to add the dimensionality reduction parameters and the pooling mode"""

    model_type = "mpnet_reduced"

    def __init__(self, *args, pooling_mode="mean", **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_mode = pooling_mode