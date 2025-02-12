# configuration_qwen2_reduced.py

from transformers import Qwen2Config
from ...configuration_reduced import ReducedConfig

class Qwen2ReducedConfig(ReducedConfig, Qwen2Config):
    """Wrapper for Qwen2Config to add dimensionality reduction parameters."""

    model_type = "qwen2_reduced"

    def __init__(
        self, 
        *args, 
        reduction_sizes = [3584,2048,1536,1024,768,512,384,256,128,64], 
        pooling_mode = "last",
        **kwargs
    ):
        super().__init__(*args, reduction_sizes=reduction_sizes, **kwargs)
        self.pooling_mode = pooling_mode
        