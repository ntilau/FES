"""Convert radians to degrees"""
import numpy as np


def rad2deg(rad):
    """
    Convert radians to degrees.
    
    Parameters
    ----------
    rad : float or array_like
        Angle in radians
        
    Returns
    -------
    deg : float or ndarray
        Angle in degrees
    """
    return (180 / np.pi) * rad
