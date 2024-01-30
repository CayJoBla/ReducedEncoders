# modeling_mpnet_reduced.py

from transformers import MPNetModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.nn.functional import mse_loss

from ...modeling_reduced import DimReduce, ReducedPreTrainedModel
from ...modeling_outputs import ReducedModelOutputWithPooling
from .modeling_sbert import SBertPooler
from .configuration_mpnet_reduced import MPNetReducedConfig


class MPNetReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced MPNet models."""
    config_class = MPNetReducedConfig
    base_model_prefix = "mpnet" # TODO: Determine whether this needs to change ("sbert"?)


class MPNetCompressedForPretraining(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SBertPooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)

    def _get_similarities(self, embeddings):
        """Returned the flattened upper triangular cosine similarity matrix of the given embeddings."""
        cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
        similarity_matrix = cos_sim(embeddings.unsqueeze(0), embeddings.unsqueeze(1))
        indices = torch.triu_indices(*similarity_matrix.shape, offset=1)
        return similarity_matrix[indices[0], indices[1]]
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        reduced_pooled = self.reduce(pooled_output)

        # TODO: There are ways to compute the loss at each layer of the reduction, is that something possible/something we want to do?

        # Compute contrastive loss
        full_similarity = self._get_similarities(pooled_output)
        reduced_similarity = self._get_similarities(reduced_pooled)
        contrastive_loss = mse_loss(full_similarity, reduced_similarity)
        print(full_similarity)
        print(reduced_similarity)

        # Compute reconstruction loss
        # TODO: Decide whether to implement this loss
        reconstruction_loss = 0     

        # Compute total loss
        loss = contrastive_loss + reconstruction_loss

        if not return_dict:
            return (embeddings, pooled_embeddings) + outputs[2:]

        return CompressedModelForPreTrainingOutput(
            loss=loss,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class SBertMPNetReducedModel(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.sbert = MPNetModel(self.config, **kwargs) if base_model is None else base_model
        self.pooler = SBertPooler(self.config)
        self.reduce = DimReduce(self.config) if reduce_module is None else reduce_module
        
        
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


class SBertMPNetReducedForSequenceClassification(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.sbert = MPNetModel(self.config, **kwargs) if base_model is None else base_model
        self.pooler = SBertPooler(self.config)
        self.reduce = DimReduce(self.config) if reduce_module is None else reduce_module
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.reduced_size, self.config.num_labels)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                labels=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
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
        pooler_output = self.pooler(sequence_output, attention_mask)   
        reduced_output = self.reduce(pooler_output)
        logits = self.classifier(self.dropout(reduced_output))

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