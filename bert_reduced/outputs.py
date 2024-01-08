from transformers.utils import ModelOutput
import torch

from typing import Optional, Tuple

@dataclass
class ReducedModelOutput(ModelOutput):
    """
    Base class for reduced model's outputs, with potential hidden states and attentions.
    Args:
        reduced_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, reduced_size)`):
            Sequence of reduced dimensionality hidden-states resulting from reducing the dimensionality of the 
            last hidden state of the model at the output of the last layer.

        Other args from transformers.modeling_outputs.BaseModelOutput
    """
    reduced_last_hidden_state: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ReducedModelOutputWithPooling(ModelOutput):
    """
    Base class for reduced model's outputs that also contains a reduced pooling of the last hidden states.

    Args:
        reduced_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, reduced_size)`):
            Sequence of reduced dimensionality hidden-states resulting from reducing the dimensionality of the 
            last hidden state of the model at the output of the last layer.
        reduced_pooler_output (`torch.FloatTensor` of shape `(batch_size, reduced_size)`):
            Pooled reduced dimensionality hidden-state resulting from reducing the dimensionality of the 
            pooler output of the model at the output of the last layer.
            
        Other args from transformers.modeling_outputs.BaseModelOutputWithPooling
    """
    reduced_last_hidden_state: torch.FloatTensor = None
    reduced_pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ReducedModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for reduced model's outputs, with potential hidden states and attentions.
    Args:
        reduced_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, reduced_size)`):
            Sequence of reduced dimensionality hidden-states resulting from reducing the dimensionality of the 
            last hidden state of the model at the output of the last layer.
        reduced_pooler_output (`torch.FloatTensor` of shape `(batch_size, reduced_size)`):
            Pooled reduced dimensionality hidden-state resulting from reducing the dimensionality of the 
            pooler output of the model at the output of the last layer.
            
        Other args from transformers.modeling_outputs.BaseModelOutputWithPoolingAndCrossAttentions
    """
    reduced_last_hidden_state: torch.FloatTensor = None
    reduced_pooler_output: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
