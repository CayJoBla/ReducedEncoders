# configuration_matryoshka.py

from transformers import MPNetConfig

class MatryoshkaConfig(MPNetConfig):
    """Wrapper for MPNetConfig for Matryoshka."""

    model_type = "matryoshka"

    def __init__(self, *args, pooling_mode="mean", matryoshka_dim=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pooling_mode = pooling_mode
        self.matryoshka_dim = matryoshka_dim or self.hidden_size