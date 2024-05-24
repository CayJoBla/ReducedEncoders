# modeling_mpnet_reduced.py

import torch
from torch import nn
from torch.nn.functional import mse_loss
from transformers import MPNetModel
from transformers.utils import ModelOutput
from dataclasses import dataclass
from typing import Optional, Tuple

from ...modeling_reduced import DimReduce, ReducedPreTrainedModel, DimExpand
from ...modeling_utils import (
    SentencePooler,
    SequenceClassificationLoss,
    ContrastiveCosineLoss,
    ReconstructionLoss,
)
from ...modeling_outputs import (
    BaseReducedModelOutput,
    BaseReducedModelOutputWithPooling, 
    ReducedSequenceClassifierOutput,
)
from .configuration_mpnet_reduced import MPNetReducedConfig


class MPNetReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced MPNet models."""
    config_class = MPNetReducedConfig
    base_model_prefix = "mpnet"


@dataclass
class MPNetCompressedForPreTrainingOutput(ModelOutput):
    """
    Ouput type of ['MPNetCompressedForPretraining'].

    Args:
        loss (torch.FloatTensor): 
            Linear combination of the contrastive learning and the reconstruction loss. 
        contrastive_loss (torch.FloatTensor, *optional*):  
            MSE between the cosine similarity of data pairs in the original and reduced embeddings. 
        reconstruction_loss (torch.FloatTensor, *optional*):
            MSE between the original and reconstructed embeddings.
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
        reconstructed_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate hidden states of the model at the output 
            of each expansion layer of the reconstruction module.
        full_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to hidden states of the model at the output of each encoder
            layer plus the optional initial embedding outputs.
        attentions (tuple(torch.FloatTensor), *optional*):
            See the documentation in `transformers.modeling_outputs.BaseModelOutput`
    """
    loss: torch.FloatTensor = None
    contrastive_loss: torch.FloatTensor = None
    reconstruction_loss: torch.FloatTensor = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reconstructed_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    full_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    

@dataclass
class MPNetCompressedModelOutput(ModelOutput):
    """
    Ouput type of ['MPNetCompressedModel'].

    Args:
        last_reduced_hidden_state (torch.FloatTensor):
            The reduced dimensionality hidden state at the output of the last layer of the reduction.
            Note that for this compressed model, this is a pooled output.
        last_full_hidden_state (torch.FloatTensor):
            The pooled hidden state at the output of the last layer of the base encoder model.
            Note that for this compressed model, this is a pooled output.
        last_reconstructed_hidden_state (torch.FloatTensor, *optional*):
            The reconstructed hidden state at the output of the last layer of the reconstruction.
            Note that for this compressed model, this is a pooled output.
        reduced_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate reduced hidden states of the model at 
            the output of each reduction layer.
            Note that for this compressed model, these are pooled outputs.
        reconstructed_hidden_states (tuple(torch.FloatTensor), *optional*):
            Tuple of tensors corresponding to the intermediate hidden states of the model at the output 
            of each expansion layer of the reconstruction module.
            Note that for this compressed model, these are pooled outputs.
    """
    last_reduced_hidden_state: torch.FloatTensor
    last_full_hidden_state: torch.FloatTensor
    last_reconstructed_hidden_state: Optional[torch.FloatTensor] = None
    reduced_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    reconstructed_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MPNetCompressedModel(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        # Construct the model
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.expand = DimExpand(self.config)    # Not needed, but included if we want to use it later 

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        full_sequence_output = full_outputs[0]
        full_pooled_output = self.pooler(full_sequence_output, attention_mask)  
        reduced_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_hidden_states[-1]

        if not return_dict:
            outputs = (reduced_pooled_output, full_pooled_output)
            if output_hidden_states:
                outputs += (reduced_hidden_states,)
            return outputs + full_outputs[2:]

        return BaseReducedModelOutput(
            last_reduced_hidden_state=reduced_pooled_output,
            last_full_hidden_state=full_pooled_output,
            reduced_hidden_states=reduced_hidden_states if output_hidden_states else None,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )


class MPNetCompressedForPreTraining(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, do_contrast=True, 
                 do_reconstruction=True, **kwargs):
        super().__init__(config)

        # Set up the losses
        if not do_contrast and not do_reconstruction:
            raise ValueError("At least one of do_contrast and do_reconstruction must be True")
        self.do_contrast = do_contrast
        self.do_reconstruction = do_reconstruction

        # Construct the model
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.expand = DimExpand(self.config) if do_reconstruction else None
        
        # Set up the loss weight parameters (only train if both losses are enabled)
        train_params = (do_contrast and do_reconstruction)
        self.params = nn.ParameterDict({
            'contrastive_weight': nn.Parameter(torch.tensor(1.), requires_grad=train_params),
            'reconstruction_weight': nn.Parameter(torch.tensor(1.), requires_grad=train_params),
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

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, 
                head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, 
                return_dict=None, compute_full_loss=True, return_loss=True):
                # Need return_loss parameter with default as True to let trainer return evaluation loss
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        f_outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        f_pooled = self.pooler(f_outputs.last_hidden_state, attention_mask)  

        r_hidden_states = self.reduce(f_pooled)
        r_pooled = r_hidden_states[-1]

        # TODO: Rethink contrastive loss -- As dimensionality increases, the angle between vectors is 
        #       more likely to be 90 degrees. Couple of issues:
        #        * Does this make the loss less meaningful?
        #        * Does the difference in dimensionality make comparing between dimensions uninformative?
        #       Maybe look into the distribution of cosine similarities in BertReduced and MPNet embeddings?
        if return_loss:
            # Compute contrastive loss
            if self.do_contrast:
                r_to_contrast = r_hidden_states if compute_full_loss else (r_pooled,)
                contrastive_loss = ContrastiveCosineLoss()(r_to_contrast, f_pooled)
            else:
                contrastive_loss = None

            # Compute reconstruction loss
            if self.do_reconstruction:
                # Pass the reduced sentence embeddings through the reconstruction module
                e_hidden_states = self.expand(r_pooled)
                e_pooled = e_hidden_states[-1]

                f_to_reconstruct = (r_hidden_states[-2::-1] + (f_pooled,)) if compute_full_loss else (f_pooled,)
                e_as_reconstructed = e_hidden_states if compute_full_loss else (e_pooled,)
                reconstruction_loss = ReconstructionLoss()(e_as_reconstructed, f_to_reconstruct)
            else:
                reconstruction_loss = None
                e_hidden_states = None

            # Compute the total weighted loss
            if self.do_contrast and self.do_reconstruction:
                # Add a homoscedastic regularization term (w1 L1 + w2 L2 - 1/2 log(w1 w2)) 
                # This is to help train the alpha and beta los weight parameters and balance the two losses
                weighted_contrastive_loss = self.params.contrastive_weight * contrastive_loss
                weighted_reconstruction_loss = self.params.reconstruction_weight * reconstruction_loss
                regularization_term = -.5*torch.log(self.params.contrastive_weight * self.params.reconstruction_weight)
                total_weighted_loss = weighted_contrastive_loss + weighted_reconstruction_loss + regularization_term
            elif self.do_contrast:      # If only one of the losses is enabled, just use that loss
                total_weighted_loss = contrastive_loss
            elif self.do_reconstruction:
                total_weighted_loss = reconstruction_loss

        else:
            contrastive_loss, reconstruction_loss, total_weighted_loss = None, None, None

        if not return_dict:
            outputs = (total_weighted_loss, contrastive_loss, reconstruction_loss) 
            if output_hidden_states:
                outputs += (r_hidden_states, e_hidden_states)
            return outputs + f_outputs[2:]

        return MPNetCompressedForPreTrainingOutput(
            loss=total_weighted_loss,
            contrastive_loss=contrastive_loss,
            reconstruction_loss=reconstruction_loss,
            reduced_hidden_states=r_hidden_states if output_hidden_states else None,
            reconstructed_hidden_states=e_hidden_states if output_hidden_states else None,
            full_hidden_states=f_outputs.hidden_states,
            attentions=f_outputs.attentions,
        )


class MPNetCompressedForSequenceClassification(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        # Construct the model
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.layernorm = nn.LayerNorm(self.config.reduced_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.reduced_size, self.config.num_labels)
        self.seq_class_loss = SequenceClassificationLoss(self.config)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                labels=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        full_sequence_output = full_outputs[0]
        full_pooled_output = self.pooler(full_sequence_output, attention_mask)  
        reduced_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_hidden_states[-1]

        pooled_output = self.dropout(self.layernorm(reduced_pooled_output)) # Add layer norm and dropout for non-linearity
        logits = self.classifier(pooled_output)
        loss = self.seq_class_loss(logits, labels) if labels is not None else None

        if not return_dict:
            outputs = (loss, logits) if loss is not None else (logits,)
            if output_hidden_states:
                outputs += (reduced_hidden_states,)
            return outputs + full_outputs[2:]

        return ReducedSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            reduced_hidden_states=reduced_hidden_states if output_hidden_states else None,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )


class MPNetReducedModel(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = MPNetModel(self.config, **kwargs) if base_model is None else base_model
        self.pooler = SentencePooler(self.config)
        self.reduce = DimReduce(self.config) if reduce_module is None else reduce_module
        
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        full_sequence_output = full_outputs[0]
        full_pooled_output = self.pooler(full_sequence_output, attention_mask)  

        reduced_sequence_hidden_states = self.reduce(full_sequence_output)
        reduced_sequence_output = reduced_sequence_hidden_states[-1]
        reduced_pooled_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_pooled_hidden_states[-1]

        reduced_hidden_states = (reduced_sequence_hidden_states, reduced_pooled_hidden_states) if output_hidden_states else None

        if not return_dict:
            outputs = (reduced_sequence_output, reduced_pooled_output, full_sequence_output, full_pooled_output)
            if output_hidden_states:
                outputs += (reduced_hidden_states,)
            return outputs + full_outputs[2:]

        return BaseReducedModelOutputWithPooling(
            last_reduced_hidden_state=reduced_sequence_output,
            last_full_hidden_state=full_sequence_output,
            reduced_pooler_output=reduced_pooled_output,
            full_pooler_output=full_pooled_output,
            reduced_hidden_states=reduced_hidden_states,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )


class MPNetReducedForSequenceClassification(MPNetReducedPreTrainedModel):
    def __init__(self, config=None, base_model=None, reduce_module=None, **kwargs):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = MPNetModel(self.config, **kwargs) if base_model is None else base_model
        self.pooler = SentencePooler(self.config)
        self.reduce = DimReduce(self.config) if reduce_module is None else reduce_module
        self.layernorm = nn.LayerNorm(self.config.reduced_size, eps=self.config.layer_norm_eps)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = nn.Linear(self.config.reduced_size, self.config.num_labels)
        self.seq_class_loss = SequenceClassificationLoss(self.config)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                labels=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        full_outputs = self.mpnet(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        full_sequence_output = full_outputs[0]
        full_pooled_output = self.pooler(full_sequence_output, attention_mask)

        reduced_pooled_hidden_states = self.reduce(full_pooled_output)
        reduced_pooled_output = reduced_pooled_hidden_states[-1]

        pooled_output = self.dropout(self.layernorm(reduced_pooled_output)) # Add layer norm and dropout for non-linearity
        logits = self.classifier(pooled_output)
        loss = self.seq_class_loss(logits, labels) if labels is not None else None

        if not return_dict:
            outputs = (loss, logits) if loss is not None else (logits,) 
            if output_hidden_states:
                outputs += (reduced_hidden_states,)
            return outputs + full_outputs[2:]

        return ReducedSequenceClassifierOutput(
            loss=loss,
            logits=logits,
            reduced_hidden_states=reduced_pooled_hidden_states if output_hidden_states else None,
            full_hidden_states=full_outputs.hidden_states,
            attentions=full_outputs.attentions,
        )