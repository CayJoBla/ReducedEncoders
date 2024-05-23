import torch
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers.utils import ModelOutput

# TODO: Update the Reduced models to use these new outputs
@dataclass
class BaseReducedModelOutput(ModelOutput):
    """
    Base class for reduced model outputs.

    Args:
        last_reduced_hidden_state (torch.FloatTensor):
            The reduced dimensionality hidden states at the output of the last layer of the model.
        last_full_hidden_state (torch.FloatTensor):
            The hidden states at the output of the last layer of the base encoder model.
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors of attentions weights after the attention softmax, used to compute the 
            weighted average in the self-attention heads.
    """
    last_reduced_hidden_state: torch.FloatTensor = None
    last_full_hidden_state: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseReducedModelOutputWithPooling(ModelOutput):
    """
    Base class for reduced model outputs with pooling.

    Args:
        last_reduced_hidden_state (torch.FloatTensor):
            The reduced dimensionality hidden states at the output of the last layer of the model.
        last_full_hidden_state (torch.FloatTensor):
            The hidden states at the output of the last layer of the base encoder model.
        reduced_pooler_output (torch.FloatTensor):
            A reduced-dimensionality hidden state pooled from the last_reduced_hidden_state sequence
        full_pooler_output (torch.FloatTensor):
            A hidden state pooled from the last_full_hidden_state sequence
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.modeling_outputs.BaseModelOutput`
    """
    last_reduced_hidden_state: torch.FloatTensor = None
    last_full_hidden_state: torch.FloatTensor = None
    reduced_pooler_output: torch.FloatTensor = None
    full_pooler_output: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseReducedModelOutputWithPoolingAndCrossAttentions(ModelOutput):
    """
    Base class for reduced model outputs with pooling and cross-attentions.

    Args:
        last_reduced_hidden_state (torch.FloatTensor):
            The reduced dimensionality hidden states at the output of the last layer of the model.
        last_full_hidden_state (torch.FloatTensor):
            The hidden states at the output of the last layer of the base encoder model.
        reduced_pooler_output (torch.FloatTensor):
            A reduced-dimensionality hidden state pooled from the last_reduced_hidden_state sequence
        full_pooler_output (torch.FloatTensor):
            A hidden state pooled from the last_full_hidden_state sequence
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.modeling_outputs.BaseModelOutput`
        cross_attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.modeling_outputs.BaseModelOutputWithCrossAttentions`
        past_key_values (tuple(tuple(torch.FloatTensor)), *optional*):
            See the documentation in `transformers.modeling_outputs.BaseModelOutputWithPast`
    """
    last_reduced_hidden_state: torch.FloatTensor = None
    last_full_hidden_state: torch.FloatTensor = None
    reduced_pooler_output: torch.FloatTensor = None
    full_pooler_output: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class ReducedSequenceClassifierOutput(ModelOutput):
    """
    Base class for outputs of sentence classification models with dimensionality reduction.

    Args:
        loss (torch.FloatTensor, *optional*):
            See the documentation in `transformers.modeling_outputs.SequenceClassifierOutput`
        logits (torch.FloatTensor):
            See the documentation in `transformers.modeling_outputs.SequenceClassifierOutput`
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.modeling_outputs.SequenceClassifierOutput`
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None