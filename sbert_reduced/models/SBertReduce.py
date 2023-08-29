# SBertReduce.py

from torch import nn
from collections import OrderedDict
import os
import json

from sentence_transformers.util import import_from_string
from .SBertReductionLayer import SBertReductionLayer
from bert_reduced import BertReduce

class SBertReduce(nn.Sequential):
    """
    Sentence transformer version of the transformers BertReduce module.

    Args:
        hidden_size (int): The size of the hidden layer (output of the BERT model) if no
            modules are provided.
        output_size (int): The size of the output later (the dimension to reduce to) if no
            modules are provided.
        inter_sizes (tuple): A sequence of intermediate layer sizes to reduce the
            dimensionality over multiple linear layers.
        modules (OrderedDict): A dictionary of modules to be used in sequence for reduction.
    """
    def __init__(self, hidden_size=768, output_size=48, inter_sizes=(), modules=None, **kwargs):
        # Create modules if not provided
        if modules is None:
            modules = OrderedDict()
            input_size = hidden_size
            for i, inter_size in enumerate(inter_sizes):   
                modules[str(i)] = SBertReductionLayer(input_size, inter_size, **kwargs)
                input_size = inter_size
            modules[str(i+1)] = SBertReductionLayer(input_size, output_size, **kwargs)
        elif not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])

        super().__init__(modules)
    
    def save(self, output_path):
        modules_config = []
        for idx, name in enumerate(self._modules):
            module = self._modules[name]
            model_path = os.path.join(output_path, str(idx)+"_"+type(module).__name__)

            os.makedirs(model_path, exist_ok=True)
            module.save(model_path)
            modules_config.append({'idx': idx, 'name': str(idx), 'path': os.path.basename(model_path), 'type': type(module).__module__})

        with open(os.path.join(output_path, 'modules.json'), 'w') as fOut:
            json.dump(modules_config, fOut, indent=2)

    @staticmethod
    def load(input_path):
        modules_json_path = os.path.join(input_path, 'modules.json')
        with open(modules_json_path) as fIn:
            modules_config = json.load(fIn)
            
        modules = OrderedDict()
        for module_config in modules_config:
            module_class = import_from_string(module_config['type'])
            module = module_class.load(os.path.join(input_path, module_config['path']))
            modules[module_config['name']] = module
        
        return SBertReduce(modules=modules)