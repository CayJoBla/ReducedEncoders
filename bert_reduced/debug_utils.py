import torch

def compare_weights(model1, model2):
    """Check whether the weights of two models are the same. Useful for determining 
    whether model weights match the base of a reduced model or whether loading scripts 
    are working properly.

    Args:
        model1 (nn.Module): The first model to compare.
        model2 (nn.Module): The second model to compare.

    Returns:    
        (bool[, int]): Whether all the weights of the two models are the same. If 
            False, the index of the first offending weight is also returned.
    """
    for i, (param1, param2) in enumerate(zip(model1.parameters(), model2.parameters())):
        if not torch.allclose(param1.data, param2.data):
            return False, i
    
    return True