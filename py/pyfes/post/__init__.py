# Lazy import — pyvista is an optional dependency
def plot_field(*args, **kwargs):
    from .plot import plot_field as _f
    return _f(*args, **kwargs)

def plot_electric_field(*args, **kwargs):
    from .plot import plot_electric_field as _f
    return _f(*args, **kwargs)

def plot_magnetic_field(*args, **kwargs):
    from .plot import plot_magnetic_field as _f
    return _f(*args, **kwargs)

def plot_mesh(*args, **kwargs):
    from .plot import plot_mesh as _f
    return _f(*args, **kwargs)
