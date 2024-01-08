# SBertMPNetReducedModel.py

from transformers import MPNetModel
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .MPNetReducedPreTrainedModel import MPNetReducedPreTrainedModel
from .SBertPooler import SBertPooler
from .DimReduce import DimReduce

class SBertMPNetReducedModel(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        self._initialize_config(config)
        super().__init__(self.config)

        self.sbert = MPNetModel(self.config, **kwargs) if base_model is None else base_model
        self.pooler = SBertPooler(self.config)
        self.reduce = DimReduce(self.config) if reduce_module is None else reduce_module
        
    def _initialize_config(self, config=None, reduction_sizes=(48,), pooling_mode="mean"):
        super()._initialize_config(config=config, reduction_sizes=reduction_sizes)
        if not hasattr(self.config, 'pooling_mode'):
            config.pooling_mode = pooling_mode
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.sbert(
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
        reduced_seq, reduced_pooled = self.reduce(sequence_output), self.reduce(pooled_output) 

        if not return_dict:
            return (embeddings, pooled_embeddings) + outputs[2:]

        return ReducedModelOutputWithPooling(
            last_hidden_state=reduced_seq,
            pooler_output=reduced_pooled,
            unreduced_last_hidden_state=sequence_output,
            unreduced_pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, reduce_module=None, 
                        base_model_class=None, **kwargs):
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, 
                                        reduce_module=reduce_module, base_model_class=base_model_class, 
                                        **kwargs)