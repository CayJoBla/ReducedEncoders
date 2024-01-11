# load_utils.py

from transformers import PreTrainedModel, AutoConfig
from .modeling_reduce import DimReduce

class DimReduceLoader(PreTrainedModel):
    """A wrapper for the DimReduce module to load the pretrained reduction weights from a 
    Huggingface hub model. This module is not meant to be run or included in a model."""
    # These class values must remain unassigned so that the base_model_prefix is not used
    config_class = None
    base_model_prefix = ""

    def __init__(self, config):
        super().__init__(config)
        self.reduce = DimReduce(config)

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        config = kwargs.pop('config', AutoConfig.from_pretrained(*args, **kwargs))
        return super().from_pretrained(*args, config=config, **kwargs).reduce