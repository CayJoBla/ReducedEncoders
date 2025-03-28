import mteb
from mteb.encoder_interface import PromptType
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import nn as nn

from reduced_encoders import MPNetReducedModel


class ReduceWrapper(nn.Module):
    def __init__(self, reduce):
        super(ReduceWrapper, self).__init__()
        self.reduce = reduce

    def forward(self, features):
        features["sentence_embedding"] = self.reduce(features["sentence_embedding"])
        return features

class ReducedSentenceTransformer(SentenceTransformer):
    def __init__(self, model_name, reduce_module):
        super().__init__(model_name)
        self.reduce = reduce_module

    def forward(self, input, **kwargs):
        if self.module_kwargs is None:
            return super().forward(input)

        for module_name, module in self.named_children():
            if module_name == "2":  # Skip Normalize
                # print("Skipping:", module)
                continue
            module_kwarg_keys = self.module_kwargs.get(module_name, [])
            module_kwargs = {key: value for key, value in kwargs.items() if key in module_kwarg_keys}
            input = module(input, **module_kwargs)
        return input

# model_name = "sentence-transformers/all-mpnet-base-v2"
# model = SentenceTransformer(model_name)
base_checkpoint = "sentence-transformers/all-mpnet-base-v2"
reduce_checkpoint = "cayjobla/all-mpnet-base-v2-compressed"
reduce = MPNetReducedModel.from_pretrained(reduce_checkpoint).reduce
model = ReducedSentenceTransformer(base_checkpoint, ReduceWrapper(reduce))
model_name = reduce_checkpoint

benchmark = mteb.get_benchmark("MTEB(eng, v2)")
evaluation = mteb.MTEB(tasks=benchmark)
results = evaluation.run(model, output_folder=f"results/{model_name}")
