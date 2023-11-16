# SBertReductionLayer.py

import torch
from torch import nn
import os
import json

from sentence_transformers.models import Dense, Dropout
from sentence_transformers.util import import_from_string

class SBertReductionLayer(nn.Module):
    """
    Layer of the SBertReduce module. Includes a dense layer, an activation function, 
    and a dropout layer.
    """
    def __init__(self, in_features, out_features, activation_function=nn.Tanh(), dropout=0.1, **kwargs):
        super().__init__()
        self.dense = Dense(in_features, out_features, activation_function=activation_function, **kwargs)
        self.dropout = Dropout(dropout)
        
    def forward(self, x):
        if "sentence_embedding" not in x:
            x['sentence_embedding'] = x['token_embeddings']
        output = self.dense(x)
        output = self.dropout(output)
        return output
    
    def get_config_dict(self):
        self.config_keys = ['input_size', 'output_size']
        return {key: self.__dict__[key] for key in self.config_keys}
    
    def save(self, output_path):
        config_dict = self.dense.get_config_dict()
        config_dict.update({"dropout": self.dropout.dropout})
        
        self.dense.save(output_path)

        with open(os.path.join(output_path, 'config.json'), 'w') as fOut:
            json.dump(config_dict, fOut)

    @staticmethod
    def load(input_path):
        config_path = os.path.join(input_path, 'config.json')
        with open(config_path) as fIn:
            config = json.load(fIn)

        config['activation_function'] = import_from_string(config['activation_function'])()
        model = SBertReductionLayer(**config)

        model.dense.load_state_dict(torch.load(os.path.join(input_path, 'pytorch_model.bin')))

        return model