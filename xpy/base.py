from .backend import xp
from typing import Callable

lib = xp()

class Primitives:
    pass

def add_prim(name: str):
    global Primitives
    # Handle both simple functions and nested modules
    parts = name.split('.')
    obj = lib
    for part in parts:
        obj = getattr(obj, part)
    
    # Store with dots replaced by underscores for attribute access
    attr_name = name.replace('.', '_')
    setattr(Primitives, attr_name, staticmethod(obj))

def add_prim_with_list(names: list[str]):
    for n in names:
        add_prim(n)

def primitive(name: str):
    return getattr(Primitives, name.replace('.', '_'))


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
add_prim_with_list(elementwise_ops)
add_prim_with_list(linear_algebra_ops)
add_prim_with_list(reduction_ops)
add_prim_with_list(array_manip_ops)
# add_prim_with_list(special_ops)
add_prim_with_list(optimization_ops)

# ============ COMPOSITE IMPLEMENTATIONS ============
# Add essential composite operations that are worth having as primitives

def add_composite_primitives():
    """Add essential composite operations that benefit from JIT"""
    try:
        # Softmax - critical for attention, often optimized
        def softmax(x, axis=-1):
            x_max = lib.max(x, axis=axis, keepdims=True)
            x_safe = x - x_max
            exp_x = lib.exp(x_safe)
            return exp_x / lib.sum(exp_x, axis=axis, keepdims=True)
        
        Primitives.softmax = staticmethod(softmax)
        
        # LogSoftmax - more numerically stable
        def log_softmax(x, axis=-1):
            x_max = lib.max(x, axis=axis, keepdims=True)
            x_safe = x - x_max
            log_sum_exp = lib.log(lib.sum(lib.exp(x_safe), axis=axis, keepdims=True))
            return x_safe - log_sum_exp
        
        Primitives.log_softmax = staticmethod(log_softmax)
        
        # ReLU - simple but everywhere
        def relu(x):
            return lib.maximum(x, 0)
        
        Primitives.relu = staticmethod(relu)
        
        # GELU approximation (used in Transformers)
        def gelu(x):
            # 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x**3)))
            sqrt_2_over_pi = lib.sqrt(2 / lib.pi)
            return 0.5 * x * (1 + lib.tanh(sqrt_2_over_pi * (x + 0.044715 * x**3)))
        
        Primitives.gelu = staticmethod(gelu)
        
        # LayerNorm helper
        def normalize(x, axis=-1, eps=1e-5):
            mean = lib.mean(x, axis=axis, keepdims=True)
            var = lib.var(x, axis=axis, keepdims=True)
            return (x - mean) / lib.sqrt(var + eps)
        
        Primitives.normalize = staticmethod(normalize)
        
    except Exception as e:
        print(f"Warning: Could not add composite primitives: {e}")

add_composite_primitives()

# ============ MINIMALIST ALTERNATIVE ============
"""
If you want the ABSOLUTE minimum for a JAX-like system:

MINIMAL_PRIMITIVES = [
    # Elementwise
    'add', 'multiply', 'subtract', 'divide',
    'exp', 'log', 'sin', 'cos', 'tanh',
    'maximum', 'minimum', 'greater', 'less',
    
    # Linear algebra
    'matmul', 'dot',
    
    # Reductions
    'sum', 'max', 'min',
    
    # Array manipulation
    'reshape', 'transpose', 'concatenate',
    
    # Special
    'where',
    
    # That's it! Everything else can be built from these.
]
"""

# print(f"Total primitives loaded: {len([x for x in dir(Primitives) if not x.startswith('_')])}")


def construct(func:Callable, name:str):
    global Primitives
    setattr(Primitives, name, staticmethod(func))

