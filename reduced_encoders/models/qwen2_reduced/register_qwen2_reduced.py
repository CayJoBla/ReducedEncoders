from transformers import (
    AutoConfig, 
    AutoModel, 
    AutoModelForPreTraining, 
    AutoModelForSequenceClassification
)
from .configuration_qwen2_reduced import Qwen2ReducedConfig
from .modeling_qwen2_reduced import (
    Qwen2ReducedModel, 
    # Qwen2ReducedForSequenceClassification, 
    Qwen2CompressedForPretraining
)

AutoConfig.register(
    Qwen2ReducedConfig.model_type, Qwen2ReducedConfig
)
AutoModel.register(
    Qwen2ReducedConfig, Qwen2ReducedModel
)
AutoModelForPreTraining.register(
    Qwen2ReducedConfig, Qwen2CompressedForPretraining
)
# AutoModelForSequenceClassification.register(
#     Qwen2ReducedConfig, MPNetReducedForSequenceClassification
# )