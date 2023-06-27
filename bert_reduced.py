# bert_reduced.py

import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertOnlyMLMHead
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput


# Create a module to reduce the dimension size of the hidden state
class BertReduce(nn.Module):
    """
    Module to insert between the base BERT model and the model head in order to reduce 
    the dimensionality of the hidden states. Uses the hidden_activation parameter from
    the base BERT configuation and the custom reduced_size parameter.

    Args:
        config (BertConfig): Configuration for the BERT model. Should also include the 
            reduced_size parameter for the final size of the output of this module.
        inter_sizes (tuple): A sequence of intermediate layer sizes to reduce the
            dimensionality over multiple linear layers.
    """
    def __init__(self, config, inter_sizes=()):
        super().__init__()
        input_size = config.hidden_size
        output_size = config.reduced_size
        
        self.layer = nn.ModuleList()       # Create a ModuleList of reduction layers
        for inter_size in inter_sizes:   
            self.layer.append(BertReductionLayer(input_size, inter_size, config))
            input_size = inter_size
        self.layer.append(BertReductionLayer(input_size, output_size, config))
        
    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states
        
    
class BertReductionLayer(nn.Module):
    """
    Layer of the BertReduction module. Includes a linear layer, an activation function, 
    and a dropout layer.
    """
    def __init__(self, input_size, output_size, config):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        if isinstance(config.hidden_act, str):
            self.reduce_act_fn = ACT2FN[config.hidden_act]
        else:
            self.reduce_act_fn = config.hidden_act
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, x):
        output = self.dense(x)
        output = self.reduce_act_fn(output)
        output = self.dropout(output)
        return output
    

# Recreate the MLM model head, but with the reduced hidden state size
class BertReducedMLMHead(BertOnlyMLMHead):
    def __init__(self, config):
        # Create a copy of the config with hidden_size=reduced_size to reshape default head
        config_dict = config.to_diff_dict()
        config_dict["hidden_size"] = config_dict.pop("reduced_size")
        
        super().__init__(BertConfig.from_dict(config_dict))
    
    
class BertReducedModel(BertPreTrainedModel):
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
        default_reduction = 48
        if config is None:
            if _from_pretrained_base:
                config = BertConfig.from_pretrained(_from_pretrained_base)
                config.reduced_size = default_reduction
            else:
                config = BertConfig(reduced_size=default_reduction)
        elif type(config) == BertConfig and not hasattr(config, 'reduced_size'):
            config.reduced_size = default_reduction
            
        super().__init__(config)
        self.config = config
        
        self._load_pretrained_base(_from_pretrained_base, **kwargs)
        self.reduce = BertReduce(config, inter_sizes=inter_sizes)
        self._load_head()
        
        self.post_init()
    
    def _load_pretrained_base(self, pretrained_model_name_or_path, *args, **kwargs):
        """
        Load the weights of a pretrained BERT model into the reduced model.
        """
        config = kwargs.pop("config", self.config)
        
        if pretrained_model_name_or_path is None:
            bert = BertModel(*args, config=config, **kwargs)
        else:
            bert = BertModel.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
            
        self.bert_name = pretrained_model_name_or_path
        self.bert = bert
        
    def _load_head(self):
        """
        Load the head model for the task being done. This method should be overridden by derived class.
        """
        pass
        
    @classmethod
    def from_pretrained(cls, *args, _from_pretrained_base=None, **kwargs):
        model = super(BertReducedModel, cls).from_pretrained(*args, **kwargs)
        
        if _from_pretrained_base is not None:
            config = kwargs.pop("config", model.config)
            model._load_pretrained_base(_from_pretrained_base, *args, config=config, **kwargs)
        return model
    
    
# Create the full model
class BertReducedForMaskedLM(BertReducedModel):
    """
    The reduced BERT model for masked language modelling.
    """
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        last_hidden_state = outputs[0]        # Sequence output
        reduced_output = self.reduce(last_hidden_state)
        prediction_scores = self.cls(reduced_output)

        # Compute loss
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _load_head(self):
        self.cls = BertReducedMLMHead(self.config)
    
    def _load_pretrained_base(self, pretrained_model_name_or_path, *args, add_pooling_layer=False, **kwargs):
        super()._load_pretrained_base(pretrained_model_name_or_path, 
                                      *args, 
                                      add_pooling_layer=add_pooling_layer, 
                                      **kwargs)
        

class BertReducedForSequenceClassification(BertReducedModel):
    """
    The reduced BERT model for sequence classification.
    """ 
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooler_output = outputs[1]
        reduced_output = self.reduce(pooler_output)
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
    
    def _load_head(self):
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2
        self.classifier = nn.Linear(in_features=self.config.reduced_size, out_features=self.config.num_labels)