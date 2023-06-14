# bert_reduced_model.py

import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput


# Create a module to reduce the dimension size of the hidden state
class BertReduce(nn.Module):
    """
    Module to insert between the base BERT model and the model head in order to reduce 
    the dimensionality of the hidden states. Uses the hidden_activation parameter from
    the base BERT configuation

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
    

# Recreate the model head, but with the reduced hidden state size
class BertReducedMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertReducedLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    

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
    
    
class BertReducedPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.reduced_size, config.reduced_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.reduced_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

# Create the full model
class BertReducedForMaskedLM(BertPreTrainedModel):
    """
    The BERT model with dimensionality reduction and model head trained for masked 
    language modeling. 

    Args:
        _from_pretrained_base_model (str, BertModel): A pretrained model to load for
            BERT. If None, a new model with random weights is initialized.
    """
    def __init__(self, config=None, inter_sizes=(512,256,128,64), _from_pretrained_base_model=None):
        if _from_pretrained_base_model:
            if isinstance(_from_pretrained_base_model, str):
                if not config: config = BertConfig.from_pretrained(_from_pretrained_base_model)
                base_model = BertModel.from_pretrained(_from_pretrained_base_model,
                                                       ignore_mismatched_sizes=True,
                                                       add_pooling_layer=False,)
            elif isinstance(_from_pretrained_base_model, BertModel):
                base_model = _from_pretrained_base_model
                if not config: config = base_model.config
            else:
                raise TypeError("To use a pretrained base BERT model with reduction, the _from_pretrained_base_model" 
                                "attribute must be a string or a BertModel object")
            
        if not hasattr(config, 'reduced_size'):
            config.reduced_size = 48
            
        super().__init__(config)
        
        self.bert = base_model if _from_pretrained_base_model else BertModel(config, add_pooling_layer=False)
        self.reduce = BertReduce(config, inter_sizes=inter_sizes)
        self.cls = BertReducedMLMHead(config)
        
        self.post_init()
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None, labels=None, 
                output_attentions=None, output_hidden_states=None, return_dict=None):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        reduced_output = self.reduce(sequence_output)
        prediction_scores = self.cls(reduced_output)

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