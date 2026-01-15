from .backend import xp
from typing import Callable

lib = xp()

class NpPrimitives: pass
class CpPrimitives: pass

def add_prim(name: str):
    parts = name.split('.')

    import numpy as np
    try:
        obj = np
        for p in parts:
            obj = getattr(obj, p)
        setattr(NpPrimitives, name.replace('.', '_'), staticmethod(obj))
    except AttributeError:
        pass

    try:
        obj = lib
        for p in parts:
            obj = getattr(obj, p)
        setattr(CpPrimitives, name.replace('.', '_'), staticmethod(obj))
    except AttributeError:
        pass

def add_prim_with_list(names: list[str]):
    for n in names:
        add_prim(n)


def primitive(device: str, name: str):
    table = {'cpu': NpPrimitives, 'cuda': CpPrimitives}
    cls = table.get(device)
    if cls is None:
        raise TypeError("device must be 'cpu' or 'cuda'")
    attr = name.replace('.', '_')
    if not hasattr(cls, attr):
        raise KeyError(f"{name} not available for {device}")
    return getattr(cls, attr)

# ============ ABSOLUTE ESSENTIALS ============
# These are the core operations that every JIT system needs

# Elementwise operations
elementwise_ops = [
    'add', 'subtract', 'multiply', 'divide',
    'power', 'sqrt', 'exp', 'log', 'log1p', 'expm1',
    'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
    'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh',
    'abs', 'absolute', 'negative', 'positive',
    'floor', 'ceil', 'round', 'sign',
    'maximum', 'minimum', 'clip',
    'greater', 'greater_equal', 'less', 'less_equal',
    'equal', 'not_equal', 'logical_and', 'logical_or',
    'logical_not', 'logical_xor',
]

# Linear algebra (must-haves for ML)
linear_algebra_ops = [
    'matmul', 'dot',  # Core matrix operations
    'tensordot',      # General tensor contractions
    'transpose',      # View manipulation
    'trace', 'diag',  # Matrix properties
]

# Reductions (critical for loss functions, normalization)
reduction_ops = [
    'sum', 'mean', 'prod', 'max', 'min',
    'all', 'any',  # Boolean reductions
]

# Array manipulation (for reshaping computational graphs)
array_manip_ops = [
    'reshape', 'expand_dims', 'squeeze',
    'concatenate', 'stack', 'split',
    'take', 'put',  # Indexing/scattering
    'where',        # Conditional selection
]

# special_ops = [
#     'erf', 'erfc',  # Used in GELU, approximations
# ]

# ============ OPTIMIZATION-SPECIFIC ============
# These help with compiler optimizations
optimization_ops = [
    'broadcast_to',  # Explicit broadcasting for optimization
]

# ============ COMPOSITE BUILDING BLOCKS ============
# These are often implemented but useful to have as primitives
composite_ops = [
    'softmax',       # Actually composite but often special-cased
    'log_softmax',   # Same
    'conv',          # If available (often composite in JAX)
    'conv_transpose',
]

# Add all essentials

def funbuild():
    add_prim_with_list(elementwise_ops)
    add_prim_with_list(linear_algebra_ops)
    add_prim_with_list(reduction_ops)
    add_prim_with_list(array_manip_ops)
    add_prim_with_list(optimization_ops)

funbuild()

def construct(func:Callable, name:str):
    global CpPrimitives, NpPrimitives
    lib = xp() 
    import numpy as np

    name = lib.__name__
    ptype = CpPrimitives if name == 'cupy' else None
    if ptype is None:
        pass
    else:
        setattr(CpPrimitives, name, staticmethod(func))
    setattr(NpPrimitives, name, staticmethod(func))

