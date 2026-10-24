"""Mesh constants shared by the 2D and 3D meshers.

The values mirror the v0.1 reference so that both dimensions agree on what an
"interior" (unmarked) entity is.
"""

#: ``mesh::maxLab`` from the reference C++ (``UINT_MAX``).
MAX_LABEL = 2**32 - 1
