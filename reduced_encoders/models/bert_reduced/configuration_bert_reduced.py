# configuration_bert_reduced.py

from transformers import BertConfig, AutoConfig
from ...configuration_reduced import ReducedConfig

class BertReducedConfig(ReducedConfig, BertConfig):
    """This class wraps the BertConfig class to add the dimensionality reduction parameters."""

    model_type = "bert_reduced"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)