# AutoReducedModel.py

import warnings
from transformers import BertConfig, MPNetConfig, PretrainedConfig, AutoConfig

from .BertReducedModel import BertReducedModel
from .BertReducedForPreTraining import BertReducedForPreTraining 
from .BertReducedForSequenceClassification import BertReducedForSequenceClassification
from .SBertMPNetReducedModel import SBertMPNetReducedModel
from .SBertMPNetReducedForSequenceClassification import SBertMPNetReducedForSequenceClassification

MODEL_MAPPING = {
    BertConfig: BertReducedModel,
    MPNetConfig: SBertMPNetReducedModel,
}

MODEL_FOR_PRETRAINING_MAPPING = {
    BertConfig: BertReducedForPreTraining,
    MPNetConfig: SBertMPNetReducedModel,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING = {
    BertConfig: BertReducedForSequenceClassification,
    MPNetConfig: SBertMPNetReducedForSequenceClassification,
}

class AutoReducedModel:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            "AutoReducedModel is designed to be instantiated using the "
            "`AutoReducedModel.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoReducedModel.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        """Instantiates one of the reduced model classes from a given configuration. 
        Does not load the model weights.

        Parameters:
            config (PretrainedConfig): The configuration class of the model to load. 
                                       Supported models include:
                - BertReducedModel (BertConfig configuration class)
                - SBertMPNetReducedModel (MPNetConfig configuration class)
        """
        # Check whether the config is for a reduction model
        if "reduction_sizes" not in config.to_dict() and "reduced_size" not in config.to_dict():
            warnings.warn("The provided config does not specify reduction sizes and may not be intended for "
                          "a reduced model. Continuing with the default of a single reduction size of 48.")

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_MAPPING.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiates one of the reduced model classes from a pre-trained model configuration."""
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_MAPPING.keys())}."
        )


class AutoReducedModelForPreTraining:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            "AutoModelForPreTraining is designed to be instantiated using the "
            "`AutoModelForPreTraining.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForPreTraining.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        """Instantiates one of the reduced model classes from a given configuration. 
        Does not load the model weights.

        Parameters:
            config (PretrainedConfig): The configuration class of the model to load. 
                                       Supported models include:
                - BertReducedModelForPreTraining (BertConfig configuration class)
                - SBertMPNetReducedModel (MPNetConfig configuration class)
        """
        # Check whether the config is for a reduction model
        if "reduction_sizes" not in config.to_dict() and "reduced_size" not in config.to_dict():
            warnings.warn("The provided config does not specify reduction sizes and may not be intended for "
                          "a reduced model. Continuing with the default of a single reduction size of 48.")

        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiates one of the reduced model classes from a pre-trained model configuration."""
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_PRETRAINING_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_FOR_PRETRAINING_MAPPING.keys())}."
        )

class AutoReducedModelForSequenceClassification:
    def __init__(self, *args, **kwargs):
        raise EnvironmentError(
            "AutoModelForSequenceClassification is designed to be instantiated using the "
            "`AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelForSequenceClassification.from_config(config)` methods."
        )
    
    @classmethod
    def from_config(cls, config):
        """Instantiates one of the reduced model classes from a given configuration. 
        Does not load the model weights.

        Parameters:
            config (PretrainedConfig): The configuration class of the model to load. 
                                       Supported models include:
                - BertReducedModelForSequenceClassification (BertConfig configuration class)
                - SBertMPNetReducedModelForSequenceClassification (MPNetConfig configuration class)
        """
        # Check whether the config is for a reduction model
        if "reduction_sizes" not in config.to_dict() and "reduced_size" not in config.to_dict():
            warnings.warn("The provided config does not specify reduction sizes and may not be intended for "
                          "a reduced model. Continuing with the default of a single reduction size of 48.")

        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())}."
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """Instantiates one of the reduced model classes from a pre-trained model configuration."""
        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            f"Unrecognized configuration class {config.__class__}. Model is either not supported for this "
            f"type of AutoModel or is not a reduced model.\n"
            f"Supported configurations include: {', '.join(c.__name__ for c in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING.keys())}."
        )