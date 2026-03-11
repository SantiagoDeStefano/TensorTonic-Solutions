import numpy as np

def apply_causal_mask(scores, mask_value=-1e9):
    """
    scores: np.ndarray with shape (..., T, T)
    mask_value: float used to mask future positions (e.g., -1e9)
    Return: masked scores (same shape, dtype=float)
    """
    # Last shapes
    T = scores.shape[-1]

    # Right upper triangle
    mask = np.triu(np.ones((T, T), dtype=bool), k=1)

    scores = scores.copy()
    scores[..., mask] = mask_value
    return scores
    
    pass