"""Vector to matrix conversion"""
import numpy as np


def vector2matrix(dim, vector):
    """
    Convert a vector to a matrix of specified dimensions.
    
    The vector is reshaped into a matrix with dimensions (rows, columns).
    This is the inverse operation of matrix(:) in MATLAB.
    
    Parameters
    ----------
    dim : tuple or list
        Dimensions (rows, columns) of the output matrix
    vector : array_like
        Input vector to be reshaped into a matrix
        
    Returns
    -------
    matrix : ndarray
        Reshaped matrix of size dim
    """
    vector = np.asarray(vector).flatten()
    rows, cols = dim
    matrix = np.zeros((rows, cols))
    for j in range(cols):
        matrix[:, j] = vector[(j*rows):(j*rows + rows)]
    return matrix
