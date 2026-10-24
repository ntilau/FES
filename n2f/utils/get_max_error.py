"""Export individual error functions for backwards compatibility"""
from .error_metrics import get_l1_error, get_l2_error, get_max_error

__all__ = ['get_l1_error', 'get_l2_error', 'get_max_error']
