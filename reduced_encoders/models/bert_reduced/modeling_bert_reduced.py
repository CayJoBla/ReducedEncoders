# modeling_bert_reduced.py

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertForPreTrainingOutput
from transformers.activations import ACT2FN
from transformers import BertModel

from ...modeling_reduced import ReducedPreTrainedModel, DimReduce
from ...modeling_outputs import ReducedModelOutputWithPoolingAndCrossAttentions
from .configuration_bert_reduced import BertReducedConfig


class BertReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced BERT models."""
    config_class = BertReducedConfig
    base_model_prefix = "bert"

    def _initialize_config(self, config=None, reduction_sizes=(48,)):
        """Set the default configuration for reduced BERT models"""
        config = BertReducedConfig() if config is None else BertReducedConfig.from_config(config)
        super()._initialize_config(config, reduction_sizes=reduction_sizes)


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
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.reduced_size, eps=config.layer_norm_eps)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


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
        self._initialize_config(config)
        super().__init__(self.config)

        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)

        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output, pooled_output = outputs[:2]
        reduced_seq, reduced_pooled = self.reduce(sequence_output), self.reduce(pooled_output)

        if not return_dict:
            return (reduced_seq, reduced_pooled) + outputs[2:]

        return ReducedModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=reduced_seq,
            pooler_output=reduced_pooled,
            unreduced_last_hidden_state=sequence_output,
            unreduced_pooler_output=pooled_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
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
        self._initialize_config(config)
        super().__init__(self.config)

        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)
        self.cls = BertReducedPreTrainingHeads(self.config)

        self.post_init()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, 
                labels=None, next_sentence_label=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output, pooled_output = outputs[:2]
        reduced_seq, reduced_pooled = self.reduce(sequence_output), self.reduce(pooled_output)
        prediction_scores, seq_relationship_score = self.cls(reduced_seq, reduced_pooled)

        total_loss = None
        if labels is not None and next_sentence_label is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
        self._initialize_config(config)
        super().__init__(self.config)
        
        self.bert = base_model if base_model is not None else BertModel(self.config)
        self.reduce = reduce_module if reduce_module is not None else DimReduce(self.config)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.reduced_size, self.config.num_labels)
        
        self.post_init()

    def _initialize_config(self, config=None, reduction_sizes=(48,)):
        super()._initialize_config(config, reduction_sizes=reduction_sizes)
        if not hasattr(self.config, 'classifier_dropout') or self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None,
                labels=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
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

        pooler_output = outputs[1]
        reduced_output = self.reduce(pooler_output)
        reduced_output = self.dropout(reduced_output)
        logits = self.classifier(reduced_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.config.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.config.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.config.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

