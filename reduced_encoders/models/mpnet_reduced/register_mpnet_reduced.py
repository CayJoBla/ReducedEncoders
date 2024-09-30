from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForPreTraining, 
    AutoModelForSequenceClassification
)
from .configuration_mpnet_reduced import MPNetReducedConfig
from .modeling_mpnet_reduced import (
    MPNetReducedModel, 
    MPNetReducedForSequenceClassification, 
    MPNetCompressedForPretraining
)

AutoConfig.register(
    MPNetReducedConfig.model_type, MPNetReducedConfig
)
AutoModel.register(
    MPNetReducedConfig, MPNetReducedModel
)
AutoModelForPreTraining.register(
    MPNetReducedConfig, MPNetCompressedForPretraining
)
AutoModelForSequenceClassification.register(
    MPNetReducedConfig, MPNetReducedForSequenceClassification
)