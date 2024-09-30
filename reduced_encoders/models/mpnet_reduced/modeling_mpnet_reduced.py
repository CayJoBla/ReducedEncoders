# modeling_mpnet_reduced.py

from transformers import MPNetModel
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.nn.functional import mse_loss

from .configuration_mpnet_reduced import MPNetReducedConfig
from ...modeling_reduced import (
    ReducedPreTrainedModel, 
    DimReduce, 
    DimExpand,
)
from ...modeling_outputs import (
    ReducedModelOutputWithPooling, 
    CompressedModelForPreTrainingOutput,
)
from ...modeling_utils import (
    compressed_contrastive_loss, 
    sequence_classification_loss, 
    SentencePooler
)


class MPNetReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced MPNet models."""
    config_class = MPNetReducedConfig
    base_model_prefix = "mpnet"


class MPNetCompressedForPretraining(MPNetReducedPreTrainedModel):
    def __init__(
        self, 
        config=None,
        base_model=None,
        reduce_module=None,
        do_contrast=True,
        do_reconstruction=True,
        **kwargs
    ):
        super().__init__(config)
        config.can_reduce_sequence = False
        config.can_reduce_pooled = True

        # Set up the losses
        if not do_contrast and not do_reconstruction:
            raise ValueError("At least one of do_contrast and "
                                "do_reconstruction must be True")
        self.do_contrast = do_contrast
        self.do_reconstruction = do_reconstruction

        # Construct the model
        kwargs['add_pooling_layer'] = False     # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.expand = DimExpand(self.config) if self.do_reconstruction else None
        
        # Set up the hyperparameters
        self.params = nn.ParameterDict({
            'contrastive_weight': nn.Parameter(torch.tensor(.5), 
                                        requires_grad=do_contrast),
            'reconstruction_weight': nn.Parameter(torch.tensor(.5), 
                                        requires_grad=do_reconstruction),
        })

    def get_extra_logging_dict(self):
        """Returns a dictionary of extra parameters to log during training."""
        return {k: v.item() for k, v in self.params.items()}

    def _get_extra_loss_index_mapping(self):
        """Returns a dictionary mapping the extra losses to their indices in 
        the output tuple. 
        Note that the indices are offset by 1 to account for the `loss` output 
        being removed before the other losses are considered.
        """
        index_mapping = {}
        if self.do_contrast:
            index_mapping.update({"contrastive_loss": 0})
        if self.do_reconstruction:
            index_mapping.update({"reconstruction_loss": 1})
        return index_mapping

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None, 
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        return_loss=True,   # Allows custom Trainer to return evaluation loss
    ):
        return_dict = return_dict or self.config.use_return_dict

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
                contrastive_loss = compressed_contrastive_loss(pooled_output, 
                                                                reduced_pooled)
            # Compute reconstruction loss
            reconstruction_loss = 0
            if self.do_reconstruction:
                decoded_reduced_pooled_output = self.expand(reduced_pooled)
                reconstruction_loss = mse_loss(pooled_output, 
                                                decoded_reduced_pooled_output)
            else:
                decoded_reduced_pooled_output = None
            # Compute total loss (w1*l1 + w2*l2 - 1/2 log(w1*w2))
            weighted_contrastive_loss = \
                self.params.contrastive_weight * contrastive_loss
            weighted_reconstruction_loss = \
                self.params.reconstruction_weight * reconstruction_loss
            regularization_term = -.5*torch.log(self.params.contrastive_weight \
                                        * self.params.reconstruction_weight)
            total_weighted_loss = weighted_contrastive_loss + \
                weighted_reconstruction_loss + regularization_term
        else:
            contrastive_loss = None
            reconstruction_loss = None
            total_weighted_loss = None

        if not return_dict:
            return (
                total_weighted_loss, 
                contrastive_loss, 
                reconstruction_loss, 
                pooled_output, 
                reduced_pooled, 
                decoded_reduced_pooled_output,
            )

        return CompressedModelForPreTrainingOutput(
            loss=total_weighted_loss,
            contrastive_loss=contrastive_loss,
            reconstruction_loss=reconstruction_loss,
            pooled_output=pooled_output,
            reduced_pooled_output=reduced_pooled,
            reconstructed_pooled_output=decoded_reduced_pooled_output,
        )


class MPNetReducedModel(MPNetReducedPreTrainedModel):
    def __init__(
        self, 
        config=None, 
        base_model=None, 
        reduce_module=None, 
        **kwargs
    ):
        super().__init__(config)

        kwargs['add_pooling_layer'] = False # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        token_type_ids=None, 
        position_ids=None, 
        head_mask=None, 
        inputs_embeds=None, 
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None
    ):
        return_dict = return_dict or self.config.use_return_dict

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

        reduced_seq = None
        if self.config.can_reduce_sequence: # Reduce the word embeddings
            reduced_seq = self.reduce(sequence_output)

        reduced_pooled = None
        if self.config.can_reduce_pooled:   # Reduce the sentence embeddings
            reduced_pooled = self.reduce(pooled_output) 
        elif self.config.can_reduce_sequence:
            self.pooler(reduced_seq, attention_mask)

        if not return_dict:
            return (
                reduced_seq, 
                reduced_pooled, 
                sequence_output, 
                pooled_output, 
                outputs.hidden_states, 
                outputs.attentions
            )

        return ReducedModelOutputWithPooling(
            last_hidden_state=reduced_seq,
            pooler_output=reduced_pooled,
            unreduced_last_hidden_state=sequence_output,
            unreduced_pooler_output=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class MPNetReducedForSequenceClassification(MPNetReducedPreTrainedModel):
    def __init__(
        self, 
        config=None, 
        base_model=None, 
        reduce_module=None,
        classifier_module=None, 
        **kwargs
    ):
        super().__init__(config)

        # Initialize the config for sequence classification
        if not hasattr(self.config, 'classifier_dropout') or \
                self.config.classifier_dropout is None:
            self.config.classifier_dropout = self.config.hidden_dropout_prob
        if not hasattr(self.config, 'num_labels'):
            self.config.num_labels = 2

        kwargs['add_pooling_layer'] = False # We use our own pooling instead
        self.mpnet = base_model or MPNetModel(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        self.dropout = nn.Dropout(self.config.classifier_dropout)
        self.classifier = classifier_module or nn.Linear(
                            self.config.reduced_size, self.config.num_labels)

        if not config.can_reduce_sequence and not config.can_reduce_pooled:
            raise ValueError("The model config must have either "
                "'can_reduce_pooled' or 'can_reduce_sequence' set to True.")
                             
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None
    ):
        return_dict = return_dict or self.config.use_return_dict

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
        if self.config.can_reduce_pooled:
            pooled_output = self.pooler(sequence_output, attention_mask)   
            reduced_pooled = self.reduce(pooled_output)
        else:       # config.can_reduce_sequence = True
            reduced_seq = self.reduce(sequence_output)
            reduced_pooled = self.pooler(reduced_seq, attention_mask)
        logits = self.classifier(self.dropout(reduced_pooled))

        loss = None
        if labels is not None:
            loss = sequence_classification_loss(logits, labels, self.config)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
