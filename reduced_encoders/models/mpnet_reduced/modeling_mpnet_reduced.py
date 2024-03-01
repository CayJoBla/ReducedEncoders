# modeling_mpnet_reduced.py

from transformers import MPNetModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.nn.functional import mse_loss

from ...modeling_reduced import DimReduce, ReducedPreTrainedModel, Decoder
from ...modeling_outputs import ReducedModelOutputWithPooling, CompressedModelForPreTrainingOutput
from ...modeling_utils import compressed_contrastive_loss
from .modeling_sbert import SBertPooler
from .configuration_mpnet_reduced import MPNetReducedConfig


class MPNetReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced MPNet models."""
    config_class = MPNetReducedConfig
    base_model_prefix = "mpnet" # TODO: Determine whether this needs to change ("sbert"?)


class MPNetCompressedForPretraining(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, do_contrast=True, do_reconstruction=True, **kwargs):
        super().__init__(config)

        # Set up the losses
        if not do_contrast and not do_reconstruction:
            raise ValueError("At least one of do_contrast and do_reconstruction must be True")
        self.do_contrast = do_contrast
        self.do_reconstruction = do_reconstruction

        # Construct the model
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SBertPooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.decoder = Decoder(self.config) if self.do_reconstruction else None
        
        # Set up the hyperparameters
        self.params = nn.ParameterDict({
            'contrastive_weight': nn.Parameter(torch.tensor(.5), requires_grad=do_contrast),
            'reconstruction_weight': nn.Parameter(torch.tensor(.5), requires_grad=do_reconstruction),
        })

    def get_extra_logging_dict(self):
        """Returns a dictionary of extra parameters to log during training."""
        return {k: v.item() for k, v in self.params.items()}

    def _get_extra_loss_index_mapping(self):
        """Returns a dictionary mapping the extra losses to their indices in the output tuple. 
        Note that the indices are offset by 1 to account for the `loss` output being removed before the other losses are considered.
        """
        index_mapping = {}
        if self.do_contrast:
            index_mapping.update({"contrastive_loss": 0})
        if self.do_reconstruction:
            index_mapping.update({"reconstruction_loss": 1})
        return index_mapping

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, return_loss=True):
                    # Need return_loss parameter with default as True to let trainer return evaluation loss
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

        if return_loss:
            # Compute contrastive loss
            contrastive_loss = 0
            if self.do_contrast:
                contrastive_loss = compressed_contrastive_loss(pooled_output, reduced_pooled)

            # Compute reconstruction loss
            reconstruction_loss = 0
            if self.do_reconstruction:
                decoded_reduced_pooled_output = self.decoder(reduced_pooled)
                reconstruction_loss = mse_loss(pooled_output, decoded_reduced_pooled_output)  

            # Compute total loss (w1 L1 + w2 L2 - 1/2 log(w1 w2))
            # We add a homoscedastic regularization term to help train the alpha and beta hyperparameters
            weighted_contrastive_loss = self.params.contrastive_weight * contrastive_loss
            weighted_reconstruction_loss = self.params.reconstruction_weight * reconstruction_loss
            regularization_term = -.5*torch.log(self.params.contrastive_weight * self.params.reconstruction_weight)
            
            total_weighted_loss = weighted_contrastive_loss + weighted_reconstruction_loss + regularization_term

        else:
            contrastive_loss, reconstruction_loss, total_weighted_loss = None, None, None

        if not return_dict:
            return (total_weighted_loss, contrastive_loss, reconstruction_loss, pooled_output, reduced_pooled, 
                    decoded_reduced_pooled_output if self.do_reconstruction else None)

        return CompressedModelForPreTrainingOutput(
            loss=total_weighted_loss,
            contrastive_loss=contrastive_loss,
            reconstruction_loss=reconstruction_loss,
            pooled_output=pooled_output,
            reduced_pooled_output=reduced_pooled,
            reconstructed_pooled_output=decoded_reduced_pooled_output if self.do_reconstruction else None,
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