from transformers import AutoConfig, AutoModel, AutoModelForPreTraining, AutoModelForSequenceClassification
from .configuration_mpnet_reduced import MPNetReducedConfig
from .modeling_mpnet_reduced import (
    SBertMPNetReducedModel, 
    SBertMPNetReducedForSequenceClassification, 
    MPNetCompressedForPretraining,
    MPNetCompressedModel,
    MPNetCompressedForSequenceClassification
)

AutoConfig.register(MPNetReducedConfig.model_type, MPNetReducedConfig)
AutoModel.register(MPNetReducedConfig, SBertMPNetReducedModel)
AutoModel.register(MPNetReducedConfig, MPNetCompressedModel)
AutoModelForPreTraining.register(MPNetReducedConfig, MPNetCompressedForPretraining)
AutoModelForSequenceClassification.register(MPNetReducedConfig, SBertMPNetReducedForSequenceClassification)
AutoModelForSequenceClassification.register(MPNetReducedConfig, MPNetCompressedForSequenceClassification)