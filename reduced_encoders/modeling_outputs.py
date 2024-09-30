import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

@dataclass
class ReducedModelOutput(ModelOutput):
    """
    Reduced model output with hidden states and attentions.

    Args:
        last_hidden_state (torch.FloatTensor):
            Sequence of reduced dimensionality hidden-states at the output of 
            the last layer of the model.
        unreduced_last_hidden_state (torch.FloatTensor):
            Sequence of full-size hidden-states at the output of the last layer 
            of the base model.
        Other args from transformers.modeling_outputs.BaseModelOutput
    """
    last_hidden_state: torch.FloatTensor = None
    unreduced_last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ReducedModelOutputWithPooling(ModelOutput):
    """
    Reduced model output with hidden states, attentions, and a pooler output.

    Args:
        last_hidden_state (torch.FloatTensor):
            Sequence of reduced dimensionality hidden-states at the output of 
            the last layer of the model.
        pooler_output (torch.FloatTensor):
            Pooled reduced dimensionality hidden-states.
        unreduced_last_hidden_state (torch.FloatTensor):
            Sequence of full-size hidden-states at the output of the last layer 
            of the base model.
        unreduced_pooler_output (torch.FloatTensor):
            Pooled full-size hidden-states.
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
    Reduced model output with hidden states, attentions, pooler output, past,
    and cross attentions.

    Args:
        last_hidden_state (torch.FloatTensor):
            Sequence of reduced dimensionality hidden-states at the output of 
            the last layer of the model.
        pooler_output (torch.FloatTensor):
            Pooled reduced dimensionality hidden-states.
        unreduced_last_hidden_state (torch.FloatTensor):
            Sequence of full-size hidden-states at the output of the last layer 
            of the base model.
        unreduced_pooler_output (torch.FloatTensor):
            Pooled full-size hidden-states.
        Other args from transformers.modeling_outputs.
            BaseModelOutputWithPoolingAndCrossAttentions
    """
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    unreduced_last_hidden_state: torch.FloatTensor = None
    unreduced_pooler_output: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class CompressedModelForPreTrainingOutput(ModelOutput):
    """
    Output type of ['MPNetCompressedForPretraining']

    Args:
        loss (torch.FloatTensor): 
            Linear combination of the contrastive and reconstruction losses.
        contrastive_loss (torch.FloatTensor):
            Contrastive loss between the full_size and compressed embeddings.
        reconstruction_loss (torch.FloatTensor):
            Reconstruction loss between the full_size and reconstructed 
            embeddings.
        pooled_output (torch.FloatTensor):
            Pooled full-size hidden-states.
        reduced_pooled_output (torch.FloatTensor):
            Pooled reduced dimensionality hidden-states.
        reconstructed_pooled_output (torch.FloatTensor):
            Reconstructed pooled full-size hidden-states.
    """
    loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    reconstruction_loss: torch.FloatTensor = None
    pooled_output: torch.FloatTensor = None
    reduced_pooled_output: torch.FloatTensor = None
    reconstructed_pooled_output: torch.FloatTensor = None
    