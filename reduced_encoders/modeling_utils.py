from torch.nn.functional import mse_loss
from torch import nn
import torch

def get_cos_sim(embeddings):
    """Return the flattened upper triangular cosine similarity matrix of the 
    given embeddings.
    """
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity_matrix = cos_sim(embeddings.unsqueeze(0),embeddings.unsqueeze(1))
    indices = torch.triu_indices(*similarity_matrix.shape, offset=1)
    return similarity_matrix[indices[0], indices[1]]

def compressed_contrastive_loss(full_embeddings, reduced_embeddings):
    """Compute the contrastive loss between the full and reduced embeddings."""
    full_similarity = get_cos_sim(full_embeddings)
    reduced_similarity = get_cos_sim(reduced_embeddings)
    return mse_loss(full_similarity, reduced_similarity)

def sequence_classification_loss(logits, labels, config):
    """Compute the sequence classification loss for the given model."""
    # Determine the problem type if not specified
    if config.problem_type is None:
        if config.num_labels == 1:
            config.problem_type = "regression"
        elif config.num_labels > 1 and \
                (labels.dtype == torch.long or labels.dtype == torch.int):
            config.problem_type = "single_label_classification"
        else:
            config.problem_type = "multi_label_classification"

    # Compute the loss based on the problem type
    if config.problem_type == "regression":
        loss_fct = nn.MSELoss()
        if config.num_labels == 1:
            logits, labels = logits.squeeze(), labels.squeeze() 
    elif config.problem_type == "single_label_classification":
        loss_fct = nn.CrossEntropyLoss()
        logits, labels = logits.view(-1, config.num_labels), labels.view(-1)
    elif config.problem_type == "multi_label_classification":
        loss_fct = nn.BCEWithLogitsLoss()

    return loss_fct(logits, labels)

class SentencePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config.pooling_mode
        if self.pooling_mode not in ["mean", "max", "cls"]:
            raise ValueError(f"Unsupported pooling mode: {self.pooling_mode}")

    def forward(self, hidden_states, attention_mask):
        if self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1)
            mask = mask.expand(hidden_states.size()).float()
            return torch.sum(hidden_states * mask, 1) / torch.sum(mask, 1)
        elif self.pooling_mode == "max":
            mask = attention_mask.unsqueeze(-1)
            mask = mask.expand(hidden_states.size()).float()
            hidden_states[mask == 0] = -1e9
            return torch.max(hidden_states, 1)[0]
        elif self.pooling_mode == "cls":
            return hidden_states[:, 0]