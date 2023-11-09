# BertReducedPreTrainedModel.py

from transformers import BertPreTrainedModel, AutoConfig, PretrainedConfig, AutoModel, BertModel
import warnings

class BertReducedPreTrainedModel(BertPreTrainedModel):
    def _initialize_config(self, config, reduced_size=48):
        """Initialize the configuration of the reduced model with the `reduced_size` parameter."""
        if isinstance(config, PretrainedConfig):
            if not hasattr(config, 'reduced_size'):
                config.reduced_size = reduced_size
        else:
            raise ValueError("`config` must be a PretrainedConfig object.")
        self.config = config

    def load_reduction(self, reduction_model_name_or_path, *args, **kwargs):
        """Load the weights of a pretrained dimensionality reduction module into the reduced model."""
        reduction_config = AutoConfig.from_pretrained(reduction_model_name_or_path, *args, **kwargs)

        self.reduce = self.__class__.from_pretrained(reduction_model_name_or_path).reduce

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, inter_sizes=(512,256,128,64), 
                        reduce_module=None, reduction_model_name_or_path=None, **kwargs):
        # Load the config for the model
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        is_reduced_model = "reduced_size" in config.__dict__     # Check if this is a reduced model
        config = kwargs.pop("config", config)

        # Load the model (different depending on whether this is a reduced model or not)
        if is_reduced_model:
            model = super(BertReducedPreTrainedModel, cls).from_pretrained(pretrained_model_name_or_path, 
                                                                            *model_args, config=config, **kwargs)
        else:
            # Load the base model
            base_model = AutoModel.from_pretrained(pretrained_model_name_or_path, 
                                                    *model_args, config=config, **kwargs)

            if reduction_model_name_or_path is not None:
                reduce_module = super(BertReducedPreTrainedModel, cls).from_pretrained(
                                    reduction_model_name_or_path, 
                                    config=config
                                ).reduce

            if reduce_module is not None:
                if type(reduce_module) != BertReduce:
                    raise TypeError(f"The `reduce_module` argument must be a BertReduce object, but was a {type(reduce_module)} object.")
                model = cls(config=config, inter_sizes=inter_sizes, base_model=base_model, reduce_module=reduce_module)
            else:
                warnings.warn(f"The {cls.__name__} model is intended to have a dimensionality reduction "
                                "module, but was loaded from a pretrained model without reduction. Loading "
                                "the model with a randomly intialized reduction. To change this, either "
                                "specify the `reduction_model_name_or_path` argument in the function call, "
                                "or load a reduction separately using the `load_reduction()` method.")
                model = cls(config=config, inter_sizes=inter_sizes, base_model=base_model, reduce_module=reduce_module)
        
        return model
            