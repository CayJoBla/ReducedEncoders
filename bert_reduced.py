# bert_reduced.py

import torch
from torch import nn
from transformers import BertModel, BertConfig
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput, SequenceClassifierOutput


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
        super(BertReduce, self).__init__()
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
        super(BertReductionLayer, self).__init__()
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
class BertReducedMLMHead(nn.Module):
    def __init__(self, config):
        super(BertReducedMLMHead, self).__init__()
        self.predictions = BertReducedLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
    

class BertReducedLMPredictionHead(nn.Module):
    def __init__(self, config):
        super(BertReducedLMPredictionHead, self).__init__()
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
        super(BertReducedPredictionHeadTransform, self).__init__()
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
    

class BertReducedModel(BertPreTrainedModel):
    """
    An abstract class to implement common elements of BERT models with dimensionality reduction.

    Args:
        inter_sizes (tuple):
            the sizes of the sequence of intermediate linear layers in the dimensionality reduction
    """
    def __init__(self, config, inter_sizes=(), **kwargs):
        super(BertReducedModel, self).__init__(config)

        if not hasattr(self.config, 'reduced_size'):
            self.config.reduced_size = 48

        add_pooling_layer = kwargs.pop("add_pooling_layer", False)
        
        self.bert = BertModel(config, add_pooling_layer=add_pooling_layer, **kwargs)
        self.reduce = BertReduce(config, inter_sizes=inter_sizes)

    def from_pretrained(self, *args, ignore_mismatched_sizes=True, **kwargs):    # Default to True instead
        super(BertReducedModel, self).from_pretrained(*args, ignore_mismatched_sizes=ignore_mismatched_sizes, **kwargs)


# Create the full MLM 
class BertReducedForMaskedLM(BertReducedModel):
    """
    The reduced BERT model for masked language modeling.
    """
    def __init__(self, config, inter_sizes=(512,256,128,64)):
        # Initialize BERT and reduction layers
        super(BertReducedForMaskedLM, self).__init__(config, inter_sizes=inter_sizes, add_pooling_layer=False)

        self.cls = BertReducedMLMHead(config)   # Initialize model head
        self.post_init()                        # Initialize weights
        
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
        labels=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):
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
        logits = self.cls(reduced_output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

# Create the full model for sequence classification
class BertReducedForSequenceClassification(BertReducedModel):
    """
    The reduced BERT model for sequence classification.
    """
    def __init__(self, config, inter_sizes=(512,256,128,64)):
        # Initialize BERT and reduction layers
        super(BertReducedForSequenceClassification, self).__init__(config, inter_sizes=inter_sizes, add_pooling_layer=True)

        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        self.classifier = nn.Linear(in_features=config.reduced_size,    # Initialize model head
                                    out_features=config.num_labels)
        self.post_init()                                                # Initialize weights    
        
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
        labels=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):
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

        pooled_output = outputs[1]
        reduced_output = self.reduce(pooled_output)
        logits = self.classifier(reduced_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
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