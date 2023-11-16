# BertReducedForPreTraining.py

from torch import nn
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput
from transformers import BertModel

from .DimReduce import DimReduce
from .NewBertReducedPreTrainedModel import BertReducedPreTrainedModel
from .ModelHeads import BertReducedPreTrainingHeads

class BertReducedForPreTraining(BertReducedPreTrainedModel):
    """
    A BERT model used during pretraining. Has both an MLM and NSP head.
    
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

