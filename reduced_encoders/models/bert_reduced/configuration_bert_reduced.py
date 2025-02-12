# configuration_bert_reduced.py

from transformers import BertConfig
from ...configuration_reduced import ReducedConfig

class BertReducedConfig(ReducedConfig, BertConfig):
    """Wrapper for BertConfig to add dimensionality reduction parameters."""

    model_type = "bert_reduced"

    def __init__(
        self, 
        *args, 
        reduction_sizes = [512,256,128,68,48], 
        **kwargs
    ):
        super().__init__(*args, reduction_sizes=reduction_sizes, **kwargs)