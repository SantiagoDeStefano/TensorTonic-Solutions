import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # Write code here
    PE = np.zeros((seq_len, d_model))
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]
    div = np.exp(i * -(np.log(base)/d_model))

    PE[:, 0::2] = np.sin(pos*div)
    PE[:, 1::2] = np.cos(pos * div[:, :PE[:, 1::2].shape[1]])
    return PE
    pass