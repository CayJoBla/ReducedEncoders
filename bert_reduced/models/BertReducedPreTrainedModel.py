# BertReducedPreTrainedModel.py

from transformers import BertPreTrainedModel, BertConfig, BertModel

class BertReducedPreTrainedModel(BertPreTrainedModel):
    def _initialize_config(self, config=None, _from_pretrained_base=None, default_reduction=48):
        """Initialize the configuration of the reduced model with the `reduced_size` parameter."""
        if config is None:
            if _from_pretrained_base:
                config = BertConfig.from_pretrained(_from_pretrained_base)
                config.reduced_size = default_reduction
            else:
                config = BertConfig(reduced_size=default_reduction)
        elif type(config) == BertConfig:
            if not hasattr(config, 'reduced_size'):
                config.reduced_size = default_reduction
        else:
            raise ValueError("`config` must be a BertConfig object or NoneType.")
        self.config = config

    def _load_base_model(self, pretrained_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained BERT model into the reduced model.""" 
        config = kwargs.pop("config", self.config)
        self.bert = BertModel(config, *args, **kwargs) if pretrained_model_name_or_path is None else \
                    BertModel.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
    
    @classmethod
    def from_pretrained(cls, *args, _from_pretrained_base=None, **kwargs):
        model = super(BertReducedPreTrainedModel, cls).from_pretrained(*args, **kwargs)
        if _from_pretrained_base is not None: 
            model._load_base_model(_from_pretrained_base, *args, **kwargs)
        return model