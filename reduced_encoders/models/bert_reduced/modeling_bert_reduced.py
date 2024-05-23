# modeling_bert_reduced.py

import torch
from torch import nn
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

from ...modeling_reduced import ReducedPreTrainedModel, DimReduce
from ...modeling_utils import get_activation, SequenceClassificationLoss
from ...modeling_outputs import (
    BaseReducedModelOutputWithPoolingAndCrossAttentions, 
    ReducedSequenceClassifierOutput,
)
from .configuration_bert_reduced import BertReducedConfig


class BertReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced BERT models."""
    config_class = BertReducedConfig
    base_model_prefix = "bert"


# Recreate the BERT model heads, but with the reduced hidden state size
class BertReducedLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertReducedPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.reduced_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states
    

class BertReducedPreTrainingHeads(BertPreTrainingHeads):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertReducedLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.reduced_size, 2)
    
    
class BertReducedPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.reduced_size, config.reduced_size)
        self.transform_act_fn = get_activation(config.hidden_act)
        self.LayerNorm = nn.LayerNorm(config.reduced_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


@dataclass
class BertReducedForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertReducedForPreTraining`].

    Args:
        loss (torch.FloatTensor, *optional*):
            See the documentation in `transformers.models.bert.modeling_bert.BertForPreTrainingOutput`.
        prediction_logits (torch.FloatTensor):
            See the documentation in `transformers.models.bert.modeling_bert.BertForPreTrainingOutput`.
        seq_relationship_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            See the documentation in `transformers.models.bert.modeling_bert.BertForPreTrainingOutput`.
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.models.bert.modeling_bert.BertForPreTrainingOutput`.
    """
    loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_relationship_logits: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertReducedModel(BertReducedPreTrainedModel):
    """
    A BERT model with dimensionality reduction for getting embeddings. This model does not include a head.
    
    Args:
        config (BertConfig): Configuration for the reduced BERT model. 
        base_model: The base BERT model to use. If not specified, a new BERT model will be
            initialized using the config.
        reduce_module: The dimensionality reduction module to use. If not specified, a new
            module will be initialized using the config.
    """
    def __init__(self, config=None, base_model=None, reduce_module=None):
        super().__init__(config)

        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        full_sequence_output, full_pooled_output = full_outputs[:2]

        reduced_sequence_hidden_states = self.reduce(full_sequence_output)
        reduced_sequence_output = reduced_sequence_hidden_states[-1]
        reduced_pooled_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_pooled_hidden_states[-1]

        reduced_hidden_states = (reduced_sequence_hidden_states, reduced_pooled_hidden_states) if output_hidden_states else None

        if not return_dict:
            outputs = (reduced_sequence_output, full_sequence_output, reduced_pooled_output, full_pooled_output)
            if output_hidden_states:
                outputs += (reduced_hidden_states,)
            return outputs + full_outputs[2:]

        return BaseReducedModelOutputWithPoolingAndCrossAttentions(
            last_reduced_hidden_state=reduced_sequence_output,
            last_full_hidden_state=full_sequence_output,
            reduced_pooler_output=reduced_pooled_output,
            full_pooler_output=full_pooled_output,
            reduced_hidden_states=reduced_hidden_states,
            full_hidden_states=full_outputs.hidden_states,
            past_key_values=full_outputs.past_key_values,
            attentions=full_outputs.attentions,
            cross_attentions=full_outputs.cross_attentions,
        )


class BertReducedForPreTraining(BertReducedPreTrainedModel):
    """
    A reduced BERT model used during pretraining. Has both an MLM and NSP head.
    
    Args:
        config (BertConfig): Configuration for the reduced BERT model. 
        base_model: The base BERT model to use. If not specified, a new BERT model will be
            initialized using the config.
        reduce_module: The dimensionality reduction module to use. If not specified, a new
            module will be initialized using the config.
    """
    def __init__(self, config=None, base_model=None, reduce_module=None):
        super().__init__(config)

        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)
        self.layernorm = nn.LayerNorm(self.config.reduced_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.cls = BertReducedPreTrainingHeads(self.config)

        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, next_sentence_label=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        full_sequence_output, full_pooled_output = full_outputs[:2]

        reduced_sequence_hidden_states = self.reduce(full_sequence_output)
        reduced_sequence_output = reduced_sequence_hidden_states[-1]
        reduced_pooled_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_pooled_hidden_states[-1]

        reduced_hidden_states = (reduced_sequence_hidden_states, reduced_pooled_hidden_states) if output_hidden_states else None

        sequence_output = self.dropout(self.layernorm(reduced_sequence_output)) # Add layer norm and dropout for non-linearity
        pooled_output = self.dropout(self.layernorm(reduced_pooled_output))     
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) 
            if output_hidden_states:
                output += (reduced_sequence_hidden_states, reduced_pooled_hidden_states)
            output += full_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertReducedForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            reduced_hidden_states=reduced_hidden_states,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )


class BertReducedForSequenceClassification(BertReducedPreTrainedModel):
    """
    The reduced BERT model for sequence classification.

    Args:
        config (BertConfig): Configuration for the reduced BERT model. 
        base_model: The base BERT model to use. If not specified, a new BERT model will be
            initialized using the config.
        reduce_module: The dimensionality reduction module to use. If not specified, a new
            module will be initialized using the config.
    """
    def __init__(self, config=None, base_model=None, reduce_module=None):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2
        
        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)
        self.layernorm = nn.LayerNorm(self.config.reduced_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.reduced_size, self.config.num_labels)
        self.seq_class_loss = SequenceClassificationLoss(self.config)
        
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        full_pooled_output = full_outputs[1]

        reduced_pooled_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_pooled_hidden_states[-1]

        pooled_output = self.dropout(self.layernorm(reduced_pooled_output)) # Add layer norm and dropout for non-linearity
        logits = self.classifier(pooled_output)
        loss = self.seq_class_loss(logits, labels) if labels is not None else None

        if not return_dict:
            output = (loss, logits) if loss is not None else (logits,)
            if output_hidden_states:
                output += (reduced_pooled_hidden_states,)
            return output + full_outputs[2:]

        return ReducedSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            reduced_hidden_states=reduced_pooled_hidden_states if output_hidden_states else None,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )

