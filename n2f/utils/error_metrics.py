"""Error metrics"""
import numpy as np


def get_l1_error(matrix, ref_matrix):
    """
    Calculate the relative L1 norm error between two matrices.
    
    Parameters
    ----------
    matrix : array_like
        Matrix to compute error for
    ref_matrix : array_like
        Reference matrix
        
    Returns
    -------
    error : float
        Relative L1 norm error
    """
    matrix = np.asarray(matrix)
    ref_matrix = np.asarray(ref_matrix)
    error = np.linalg.norm(ref_matrix - matrix, ord=1) / np.linalg.norm(ref_matrix, ord=1)
    print(f'##> Relative L1 error = {error:.4g}')
    return error


def get_l2_error(matrix, ref_matrix):
    """
    Calculate the relative L2 norm error between two matrices.
    
    Parameters
    ----------
    matrix : array_like
        Matrix to compute error for
    ref_matrix : array_like
        Reference matrix
        
    Returns
    -------
    error : float
        Relative L2 norm error
    """
    matrix = np.asarray(matrix)
    ref_matrix = np.asarray(ref_matrix)
    error = np.linalg.norm(ref_matrix - matrix) / np.linalg.norm(ref_matrix)
    print(f'##> Relative L2 error = {error:.4g}')
    return error


def get_max_error(matrix, ref_matrix):
    """
    Calculate the relative max (L-infinity) norm error between two matrices.
    
    Parameters
    ----------
    matrix : array_like
        Matrix to compute error for
    ref_matrix : array_like
        Reference matrix
        
    Returns
    -------
    error : float
        Relative max norm error
    """
    matrix = np.asarray(matrix)
    ref_matrix = np.asarray(ref_matrix)
    error = np.linalg.norm(ref_matrix - matrix, ord=np.inf) / np.linalg.norm(ref_matrix, ord=np.inf)
    print(f'##> Relative max error = {error:.4g}')
    return error
