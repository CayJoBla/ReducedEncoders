# BertReducedForSequenceClassification.py

import torch
from torch import nn

from transformers.modeling_outputs import SequenceClassifierOutput

from .BertReduce import BertReduce
from .BertReducedPreTrainedModel import BertReducedPreTrainedModel

class BertReducedForSequenceClassification(BertReducedPreTrainedModel):
    """
    The reduced BERT model for sequence classification.
    """ 
    def __init__(self, config=None, inter_sizes=(512,256,128,64), _from_pretrained_base=None, **kwargs):
        self._initialize_config(config, _from_pretrained_base=_from_pretrained_base)
        super().__init__(self.config)
        
        self._load_base_model(_from_pretrained_base, **kwargs)
        self.reduce = BertReduce(self.config, inter_sizes=inter_sizes)

        classifier_dropout = (
            config.classifier_dropout if self.config.classifier_dropout is not None else self.config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self._load_head()
        
        self.post_init()

    def _load_head(self):
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2
        self.classifier = nn.Linear(in_features=self.config.reduced_size, out_features=self.config.num_labels)

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
