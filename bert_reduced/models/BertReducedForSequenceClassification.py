# BertReducedForSequenceClassification.py

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertModel

from .DimReduce import DimReduce
from .BertReducedPreTrainedModel import BertReducedPreTrainedModel

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
