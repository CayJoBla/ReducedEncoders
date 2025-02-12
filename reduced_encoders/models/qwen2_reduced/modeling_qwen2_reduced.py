# modeling_qwen_reduced.py

from transformers import Qwen2Model
from transformers.modeling_outputs import SequenceClassifierOutput
from torch import nn
import torch
from torch.nn.functional import mse_loss

from .configuration_qwen2_reduced import Qwen2ReducedConfig
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


class Qwen2ReducedPreTrainedModel(ReducedPreTrainedModel):
    """An abstract class for defining defaults for reduced BERT models."""
    config_class = Qwen2ReducedConfig
    base_model_prefix = "qwen2"
    _supports_sdpa = True


class Qwen2CompressedForPretraining(Qwen2ReducedPreTrainedModel):
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
        self.qwen2 = base_model or Qwen2Model(self.config, **kwargs)
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

        # TODO: Do I need to call self.post_init() here?

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
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None, 
        use_cache=None,
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None,
        return_loss=True,   # Allows custom Trainer to return evaluation loss
    ):
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.qwen2(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
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


class Qwen2ReducedModel(Qwen2ReducedPreTrainedModel):
    def __init__(
        self, 
        config=None, 
        base_model=None, 
        reduce_module=None, 
        **kwargs
    ):
        super().__init__(config)

        kwargs['add_pooling_layer'] = False # We use our own pooling instead
        self.qwen2 = base_model or Qwen2Model(self.config, **kwargs)
        self.pooler = SentencePooler(self.config)
        self.reduce = reduce_module or DimReduce(self.config)
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None, 
        use_cache=None,
        output_attentions=None, 
        output_hidden_states=None, 
        return_dict=None,
    ):
        return_dict = return_dict or self.config.use_return_dict

        outputs = self.qwen2(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            use_cache=use_cache,
            output_attentions=output_attentions, 
            output_hidden_states=output_hidden_states, 
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        print("Sequence output shape:", sequence_output.shape)
        pooled_output = self.pooler(sequence_output, attention_mask) 
        print("Pooled output shape:", pooled_output.shape)

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

