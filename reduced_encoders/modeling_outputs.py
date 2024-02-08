import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

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
    last_hidden_state: torch.FloatTensor = None
    unreduced_last_hidden_state: torch.FloatTensor = None
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
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    unreduced_last_hidden_state: torch.FloatTensor = None
    unreduced_pooler_output: torch.FloatTensor = None
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
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    unreduced_last_hidden_state: torch.FloatTensor = None
    unreduced_pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CompressedModelForPreTrainingOutput(ModelOutput):
    """
    Ouput type of ['MPNetCompressedForPretraining']

    Args:
        loss (torch.FloatTensor): Linear combination of the contrastive learning and the reconstruction loss. The contrastive 
            learning loss is the MSE between the cosine similarity of data pairs in the original and reduced embeddings. 
            The reconstruction loss is the MSE between the original and decoded reduced embeddings.
        hidden_states (tuple(torch.FloatTensor)): Hidden-states of the model at the output of each layer
        attentions (tuple(torch.FloatTensor)): Attentions weights after the attention softmax, used to compute the weighted 
            average in the self-attention heads.
    """
    loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    reconstruction_loss: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    reduced_pooled_output: torch.FloatTensor = None
    reconstructed_pooled_output: torch.FloatTensor = None
    