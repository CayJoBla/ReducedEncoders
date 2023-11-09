from torch import nn
import torch

class SBertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config.pooling_mode

    def forward(self, hidden_states, attention_mask):
        if self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.sum(mask, 1)
        elif self.pooling_mode == "max":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask == 0] = -1e9      # Set padding tokens to large negative value
            return torch.max(hidden_states, 1)[0]
        elif self.pooling_mode == "cls":
            return hidden_states[:, 0]
        else:
            raise NotImplementedError("Unknown pooling mode {}".format(self.pooling_mode))
        