"""
Options merging utilities for the rfm RFM implementation.

Functions for merging user-specified options with default values.
"""

from typing import Dict, Any, Optional


def merge_options(user_options: Optional[Dict[str, Any]],
                 default_options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user-specified options with default options.

    User options override default options where specified.
    Missing user options are filled in from defaults.

    Args:
        user_options: Dictionary of user-specified options (can be None)
        default_options: Dictionary of default option values

    Returns:
        Dictionary of merged options
    """
    merged = dict(default_options)
    if user_options:
        merged.update(user_options)
    return merged