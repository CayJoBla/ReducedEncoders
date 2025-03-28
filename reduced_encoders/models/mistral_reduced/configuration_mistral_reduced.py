# configuration_mistral_reduced.py

from transformers import MistralConfig  # TODO: CHECK THIS
from ...configuration_reduced import ReducedConfig

class MistralReducedConfig(ReducedConfig, MistralConfig):
    """Wrapper for MistralConfig to add dimensionality reduction parameters."""

    model_type = "mistral_reduced"

    def __init__(
        self, 
        *args, 
        reduction_sizes = [3584,2048,1536,1024,768,512,384,256,128,64], 
        pooling_mode = "last",
        **kwargs
    ):
        super().__init__(*args, reduction_sizes=reduction_sizes, **kwargs)
        self.pooling_mode = pooling_mode
        