# BertReducedModel.py

from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers import BertModel

from .DimReduce import DimReduce
from .BertReducedPreTrainedModel import BertReducedPreTrainedModel
from ..outputs import ReducedModelOutputWithPoolingAndCrossAttentions

class BertReducedModel(BertReducedPreTrainedModel):
    """
    A BERT model with dimensionality reduction for getting embeddings
    
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
            reduced_last_hidden_state=reduced_seq,
            reduced_pooler_output=reduced_pooled,
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

