from torch.nn.functional import mse_loss

def get_cos_sim(self, embeddings):
    """Returned the flattened upper triangular cosine similarity matrix of the given embeddings."""
    cos_sim = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
    similarity_matrix = cos_sim(embeddings.unsqueeze(0), embeddings.unsqueeze(1))
    indices = torch.triu_indices(*similarity_matrix.shape, offset=1)
    return similarity_matrix[indices[0], indices[1]]

def compressed_contrastive_loss(full_embeddings, reduced_embeddings):
    """Compute the contrastive loss between the full and reduced embeddings for the given model.
    This loss function is used in compressed models as an alternative to the standard BERT objective.
    """
    full_similarity = get_cos_sim(full_embeddings)
    reduced_similarity = get_cos_sim(reduced_embeddings)
    return mse_loss(full_similarity, reduced_similarity)