# modeling_matryoshka.py

from transformers import MPNetModel, MPNetConfig, PreTrainedModel, MPNetForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.nn.functional import mse_loss

from .configuration_matryoshka import MatryoshkaConfig
from ...modeling_utils import (
    sequence_classification_loss, 
    SentencePooler
)

class MatryoshkaForSequenceClassification(MPNetForSequenceClassification):
    """A Matryoshka model for sequence classification tasks."""
    config_class = MatryoshkaConfig
    base_model_prefix = "mpnet"

    def __init__(
        self, 
        config=None, 
        base_model=None, 
        reduce_module=None,
        classifier_module=None, 
        matryoshka_dim=None,
        **kwargs
    ):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or \
                self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        kwargs['add_pooling_layer'] = False # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = classifier_module or nn.Linear(
                            self.config.matryoshka_dim, self.config.num_labels)

        self.post_init()
                             
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]
        pooled_output = self.pooler(sequence_output, attention_mask)
        reduced_pooled = pooled_output[:, :self.config.matryoshka_dim]
        logits = self.classifier(self.dropout(reduced_pooled))

        loss = None
        if labels is not None:
            loss = sequence_classification_loss(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
