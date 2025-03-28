from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForPreTraining, 
    AutoModelForSequenceClassification
)
from .configuration_mistral_reduced import MistralReducedConfig
from .modeling_mistral_reduced import (
    MistralReducedModel, 
    # MistralReducedForSequenceClassification, 
    MistralCompressedForPretraining
)

AutoConfig.register(
    MistralReducedConfig.model_type, MistralReducedConfig
)
AutoModel.register(
    MistralReducedConfig, MistralReducedModel
)
AutoModelForPreTraining.register(
    MistralReducedConfig, MistralCompressedForPretraining
)
# AutoModelForSequenceClassification.register(
#     MistralReducedConfig, MPNetReducedForSequenceClassification
# )