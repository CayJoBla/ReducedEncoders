# configuration_mpnet_reduced.py

from transformers import AutoConfig, PretrainedConfig
import warnings

class ReducedEmbeddingModelConfig(PretrainedConfig):
    model_type = "reduced_embedding"

    def __init__(
        self,
        base_model_config=None,
        embedding_sizes=[],
        can_reduce_sequence=False,
        can_reduce_pooled=True,
        reduce_dropout_prob=0.1,
        reduce_activation_fn="silu",
        **kwargs
    ):
        super(ReducedEmbeddingModelConfig, self).__init__(**kwargs)

        # Initialize the base model configuration
        if isinstance(base_model_config, str):
            base_model_config = AutoConfig.from_pretrained(base_model_config)
        # elif isinstance(base_model_config, type) and issubclass(base_model_config, PretrainedConfig):
        #     base_model_config = base_model_config(**kwargs)
        if base_model_config is not None and not isinstance(base_model_config, PretrainedConfig):
            raise ValueError("base_model_config should be a model path or a PretrainedConfig")
        self.base_model_config = base_model_config

        # Set values for the reduced configuration
        self.embedding_sizes = embedding_sizes
        self.can_reduce_sequence = can_reduce_sequence
        self.can_reduce_pooled = can_reduce_pooled
        self.dropout_prob = reduce_dropout_prob
        self.activation_fn = reduce_activation_fn