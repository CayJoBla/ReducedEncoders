# configuration_bert_reduced.py

from transformers import BertConfig
from ...configuration_reduced import ReducedConfig

class BertReducedConfig(ReducedConfig, BertConfig):
    """Wrapper for BertConfig to add dimensionality reduction parameters."""

    model_type = "bert_reduced"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)