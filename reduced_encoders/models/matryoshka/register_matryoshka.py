from transformers import (
    AutoModelForSequenceClassification, 
    AutoConfig,
    AutoTokenizer,
    MPNetTokenizer,
)
from .configuration_matryoshka import MatryoshkaConfig
from .modeling_matryoshka import MatryoshkaForSequenceClassification

AutoConfig.register(
    MatryoshkaConfig.model_type, MatryoshkaConfig
)
AutoModelForSequenceClassification.register(
    MatryoshkaConfig, MatryoshkaForSequenceClassification
)
# AutoTokenizer.register(
#     MatryoshkaConfig, MPNetTokenizer
# )