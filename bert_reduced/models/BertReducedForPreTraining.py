# BertReducedForPreTraining.py

from torch import nn
from transformers.models.bert.modeling_bert import BertForPreTrainingOutput

from .BertReduce import BertReduce
from .BertReducedPreTrainedModel import BertReducedPreTrainedModel
from .ModelHeads import BertReducedPreTrainingHeads

class BertReducedForPreTraining(BertReducedPreTrainedModel):
    """
    An abstract class for defining common methods between reduced BERT models.
    
    Args:
        config (BertConfig): Configuration for the BERT model. Should also include the 
            reduced_size parameter. If not, reduced_size defaults to 48. If config=None,
            the config of _from_pretrained_base or the default BertConfig is used.
        inter_sizes (tuple): A sequence of intermediate layer sizes as defined in BertReduce
        _from_pretrained_base (str): A model name or path to a locally saved model to use as
            the base BERT model for the full reduced model. Does not load the model head.
    """
    def __init__(self, config=None, inter_sizes=(512,256,128,64), _from_pretrained_base=None, **kwargs):
        self._initialize_config(config, _from_pretrained_base=_from_pretrained_base)
        super().__init__(self.config)
        
        self._load_base_model(_from_pretrained_base, **kwargs)
        self.reduce = BertReduce(self.config, inter_sizes=inter_sizes)
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

