from .BertReducedPreTrainedModel import BertReducedPreTrainedModel
from .SBertPooler import SBertPooler
from .BertReduce import BertReduce
from transformers import BertConfig, BertModel

class SBertReducedModel(BertReducedPreTrainedModel):
    def __init__(self, config=None, inter_sizes=(512,256,128,64), base_model=None, reduce_module=None):
        # Set up model config
        if config is None: config = BertConfig()
        self._initialize_config(config)
        super().__init__(self.config)

        self.sbert = BertModel(self.config) if base_model is None else base_model
        self.pooler = SBertPooler(self.config)
        self.reduce = BertReduce(self.config, inter_sizes=inter_sizes) if reduce_module is None else reduce_module
        
    def _initialize_config(self, config=None, reduced_size=48, pooling_mode="mean"):
        """Initialize the configuration of the reduced model with the `reduced_size` and 'pooling_mode' parameters."""
        super()._initialize_config(config=config, reduced_size=reduced_size)
        if not hasattr(self.config, 'pooling_mode'):
            config.pooling_mode = pooling_mode
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None):

        outputs = self.bert(
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
        embeddings = self.pooler(sequence_output, attention_mask)   # TODO: Should the pooling and reduction be done in the other order?
        reduced_embeddings = self.reduce(pooled_output)             #       May learn more the other way...

        return (reduced_embeddings, embeddings) + outputs[2:]
