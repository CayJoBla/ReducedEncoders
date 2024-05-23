import torch.nn.functional as F
from torch import nn
import torch
from transformers.activations import ACT2FN


def get_activation(act):
    if isinstance(act, str):    # Convert string to function
        return ACT2FN[act]
    else:                       # A function is passed
        return act  

def pairwise_cosine_similarity(x):
    """Compute the pairwise cosine similarity between all rows of the input tensor x,
    and return a single vector of the flattened upper triangular cosine similarity matrix.
    """
    similarity_matrix = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
    indices = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    return similarity_matrix[*indices]


class ContrastiveCosineLoss(nn.Module):
    """Compute the contrastive cosine loss between the full and reduced embeddings for the given model.
    This loss function compares the cosine similarities of batch pairs in full and reduced embedding
    spaces, training the model to maintain the same relative distances between embeddings.

    The loss is computed as the mean squared error between the cosine similarities of the full and 
    reduced embeddings. If a sequence of reduced embeddings is provided, the loss is computed for each
    reduced embedding and summed together.
    """
    def forward(self, reduced_embeddings, full_embeddings):
        if type(reduced_embeddings) not in (list, tuple):   
            reduced_embeddings = (reduced_embeddings,)

        full_similarity = pairwise_cosine_similarity(full_embeddings)

        contrastive_loss = 0
        for embedding in reduced_embeddings:
            reduced_similarity = pairwise_cosine_similarity(embedding)
            contrastive_loss += F.mse_loss(full_similarity, reduced_similarity)
        
        return contrastive_loss


class ReconstructionLoss(nn.Module):
    """Compute the reconstruction loss between the full and reduced embeddings for the given model.
    This loss function compares the full embeddings with the reduced embeddings, training the model
    to reconstruct the full embeddings from the reduced embeddings.

    The loss is computed as the mean squared error between the full and reduced embeddings. 
    If sequences of embeddings are provided, the loss is computed for each pair of embeddings and 
    summed together.
    """
    def forward(self, reconstructed_embeddings, full_embeddings):
        if type(reconstructed_embeddings) not in (list, tuple):   
            reconstructed_embeddings = (reconstructed_embeddings,)
        if type(full_embeddings) not in (list, tuple):   
            full_embeddings = (full_embeddings,)
        if len(reconstructed_embeddings) != len(full_embeddings):
            raise ValueError("The number of full and reconstructed embeddings must match.")

        reconstruction_loss = 0
        for reconstructed, full in zip(reconstructed_embeddings, full_embeddings):
            reconstruction_loss += F.mse_loss(reconstructed, full)

        return reconstruction_loss


class SequenceClassificationLoss(nn.Module):
    """Compute the sequence classification loss for the given model.
    Takes the problem type and number of labels from the model configuration.
    Infers the problem type if not specified.
    """
    def __init__(self, config):
        super().__init__()
        self.problem_type = config.problem_type
        self.num_labels = config.num_labels

    def forward(self, logits, labels):
        # Determine the problem type if not specified
        if self.problem_type is None:
            if self.num_labels == 1:
                self.problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                self.problem_type = "single_label_classification"
            else:
                self.problem_type = "multi_label_classification"

        # Compute the loss based on the problem type
        if self.problem_type == "regression":
            loss_fct = nn.MSELoss()
            if self.num_labels == 1:
                logits, labels = logits.squeeze(), labels.squeeze()             # Format
        elif self.problem_type == "single_label_classification":
            loss_fct = nn.CrossEntropyLoss()
            logits, labels = logits.view(-1, self.num_labels), labels.view(-1)  # Format
        elif self.problem_type == "multi_label_classification":
            loss_fct = nn.BCEWithLogitsLoss()

        return loss_fct(logits, labels)


class SentencePooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pooling_mode = config.pooling_mode

    def forward(self, hidden_states, attention_mask):
        if self.pooling_mode == "mean":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            return torch.sum(hidden_states * mask, 1) / torch.sum(mask, 1)
        elif self.pooling_mode == "max":
            mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states[mask == 0] = -1e9      # Set padding tokens to large negative value
            return torch.max(hidden_states, 1)[0]
        elif self.pooling_mode == "cls":
            return hidden_states[:, 0]
        else:
            raise ValueError("Unknown pooling mode {}".format(self.pooling_mode))

    def __repr__(self):
        return f"SentencePooler(pooling_mode={self.pooling_mode})"