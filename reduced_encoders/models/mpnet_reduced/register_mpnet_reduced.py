from transformers import AutoConfig, AutoModel, AutoModelForPreTraining, AutoModelForSequenceClassification
from .configuration_mpnet_reduced import MPNetReducedConfig
from .modeling_mpnet_reduced import (
    MPNetReducedModel, 
    MPNetReducedForSequenceClassification, 
    MPNetCompressedForPreTraining,
    MPNetCompressedModel,
    MPNetCompressedForSequenceClassification
)

AutoConfig.register(MPNetReducedConfig.model_type, MPNetReducedConfig)
AutoModel.register(MPNetReducedConfig, MPNetReducedModel)
AutoModel.register(MPNetReducedConfig, MPNetCompressedModel)
AutoModelForPreTraining.register(MPNetReducedConfig, MPNetCompressedForPreTraining)
AutoModelForSequenceClassification.register(MPNetReducedConfig, MPNetReducedForSequenceClassification)
AutoModelForSequenceClassification.register(MPNetReducedConfig, MPNetCompressedForSequenceClassification)