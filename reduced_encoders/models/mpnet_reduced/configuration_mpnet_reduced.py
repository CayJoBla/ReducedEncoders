# configuration_mpnet_reduced.py

from transformers import MPNetConfig
from ...configuration_reduced import ReducedConfig

class MPNetReducedConfig(ReducedConfig, MPNetConfig):
    """Wrapper for MPNetConfig to add dimensionality reduction parameters."""

    model_type = "mpnet_reduced"

    def __init__(
        self, 
        *args, 
        reduction_sizes = [512,256,128,68,48], 
        pooling_mode = "mean", 
        **kwargs
    ):
        super().__init__(*args, reduction_sizes=reduction_sizes, **kwargs)
        self.pooling_mode = pooling_mode