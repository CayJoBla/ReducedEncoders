# bert_reduced.py

import torch
from torch import nn
from transformers import BertModel, BertConfig, BertForSequenceClassification
from transformers.activations import ACT2FN
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertPreTrainingHeads, BertForPreTrainingOutput
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
    

# Recreate the model heads, but with the reduced hidden state size
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
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
    

class BertReducedPreTrainingHeads(BertPreTrainingHeads):
    def __init__(self, config):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertReducedLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.reduced_size, 2)


class BertReducedPreTrainedModel(BertPreTrainedModel):
    def _initialize_config(self, config=None, _from_pretrained_base=None, default_reduction=48):
        """Initialize the configuration of the reduced model with the `reduced_size` parameter."""
        if config is None:
            if _from_pretrained_base:
                config = BertConfig.from_pretrained(_from_pretrained_base)
                config.reduced_size = default_reduction
            else:
                config = BertConfig(reduced_size=default_reduction)
        elif type(config) == BertConfig:
            if not hasattr(config, 'reduced_size'):
                config.reduced_size = default_reduction
        else:
            raise ValueError("`config` must be a BertConfig object or NoneType.")
        self.config = config

    def _load_base_model(self, pretrained_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained BERT model into the reduced model.""" 
        config = kwargs.pop("config", self.config)
        self.bert = BertModel(config, *args, **kwargs) if pretrained_model_name_or_path is None else \
                    BertModel.from_pretrained(pretrained_model_name_or_path, *args, config=config, **kwargs)
    
    @classmethod
    def from_pretrained(cls, *args, _from_pretrained_base=None, **kwargs):
        model = super(BertReducedPreTrainedModel, cls).from_pretrained(*args, **kwargs)
        if _from_pretrained_base is not None: 
            model._load_base_model(_from_pretrained_base, *args, **kwargs)
        return model
    
    
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
    

class SBertReduce(nn.Module):
    def __init__(self, reduce):
        super().__init__()
        self.reduce = reduce
        
    def forward(self, hidden_states):
        if type(hidden_states) == dict:
            inputs = hidden_states["token_embeddings"]
        else:
            inputs = hidden_states
            
        hidden_states.update({"token_embeddings": self.reduce(inputs)})    
        return hidden_states
    
