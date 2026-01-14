from typing import Sequence
from typing import Tuple, Union, Sequence
import uuid

class NameFiller:
    def __init__(self):
        self.counters = {}

    def get_name(self, base="var"):
        if base not in self.counters:
            self.counters[base] = 0
        name = f"{base}{self.counters[base]}"
        self.counters[base] += 1
        return name
    
name_filler = NameFiller()

class ShapeError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def broadcast_shape(shape1, shape2):
    import numpy as np
    """
    Given two shapes (tuples/lists of ints), check if they are broadcastable.
    If yes, return the resulting broadcasted shape.
    Else, raise ValueError.
    """
    # Make shapes equal length by prepending 1s
    len1, len2 = len(shape1), len(shape2)
    if len1 < len2:
        shape1 = (1,) * (len2 - len1) + tuple(shape1)
    elif len2 < len1:
        shape2 = (1,) * (len1 - len2) + tuple(shape2)

    result = []
    for dim1, dim2 in zip(shape1, shape2):
        if dim1 == dim2 or dim1 == 1 or dim2 == 1:
            result.append(max(dim1, dim2))
        else:
            raise ShapeError(f"Shapes {shape1} and {shape2} are not broadcastable.")
    return tuple(result)


def matmul_shape(shape1, shape2):
    import numpy as np
    """
    Infer the result shape of a matmul operation given two input shapes.

    """
    dummy1 = np.empty(shape1, dtype=bool)
    dummy2 = np.empty(shape2, dtype=bool)
    return (dummy1 @ dummy2).shape


def reshape_shape(input_shape, new_shape):
    def prod(a):
        x = 1 
        for i in a:
            x*= i
        return x
    input_size = prod(input_shape)
    new_shape = list(new_shape)

    # Count -1s
    neg_count = new_shape.count(-1)
    if neg_count > 1:
        raise ValueError("Only one dimension can be -1")

    # Compute product of specified dims (ignoring -1)
    known_product = 1
    for dim in new_shape:
        if dim != -1:
            known_product *= dim

    if neg_count == 1:
        if input_size % known_product != 0:
            raise ValueError("Cannot infer dimension: sizes don't match")
        # Replace -1 with inferred dimension
        for i, dim in enumerate(new_shape):
            if dim == -1:
                new_shape[i] = input_size // known_product
                break

    # Final check
    if prod(new_shape) != input_size:
        raise ValueError(f"Shape mismatch: cannot reshape {input_shape} to {tuple(new_shape)}")

    return tuple(new_shape)

def transpose_shape(shape, axes):
    import numpy as np
    dummy = np.empty(shape, dtype=bool)
    return dummy.transpose(axes).shape

def broadcast_to(data, shape):
    from .base import placeholder
    expr = f"lib.broadcast_to({data}, {shape})"
    return placeholder.place(*shape, name=expr)

def reduced_shape(
    shape: Tuple[int, ...],
    axis: Union[int, Sequence[int], None] = None,
    keepdims: bool = False
) -> Tuple[int, ...]:
    ndim = len(shape)

    # Normalize axis argument
    if axis is None:
        axes = list(range(ndim))
    elif isinstance(axis, int):
        axes = [axis % ndim]  # handle negative axis
    else:
        axes = [a % ndim for a in axis]

    if keepdims:
        # Replace reduced axes with 1
        return tuple(1 if i in axes else shape[i] for i in range(ndim))
    else:
        # Remove reduced axes
        return tuple(shape[i] for i in range(ndim) if i not in axes)

def stack_shape(shapes, axis):
    base = shapes[0]
    for s in shapes:
        if s != base:
            raise ValueError("All input shapes must match for stack")
    return base[:axis] + (len(shapes),) + base[axis:]

def pad_shape(shape, pad_width):
    if not isinstance(shape, tuple):
        raise TypeError("shape must be a tuple")

    if len(shape) != len(pad_width):
        raise ValueError(f"pad_width must have same rank as shape, "
                         f"got {len(pad_width)} vs {len(shape)}")

    new_shape = []
    for i, (dim, pad) in enumerate(zip(shape, pad_width)):
        if not (isinstance(pad, tuple) and len(pad) == 2):
            raise ValueError(f"pad_width[{i}] must be a tuple of (before, after)")
        before, after = pad
        if before < 0 or after < 0:
            raise ValueError("pad widths must be non-negative")
        new_shape.append(dim + before + after)

    return tuple(new_shape)

def max_min_shape(shape, axis=None, keepdims=False):
    ndim = len(shape)
    max_axes = list(range(ndim))

    # Case 1: axis=None
    if axis is None:
        return () if not keepdims else tuple(1 for _ in shape)

    def kdims_manager(old, reduced_axes):
        if not keepdims:
            return tuple(s for i, s in enumerate(old) if i not in reduced_axes)
        return tuple(1 if i in reduced_axes else s for i, s in enumerate(old))

    if isinstance(axis, (tuple, list)):
        axes = []
        for a in axis:
            if a < 0:
                a += ndim     
            if a < 0 or a >= ndim:
                raise ValueError(f'axis {a} is out of bounds for array of dimension {ndim}')
            axes.append(a)
        return kdims_manager(shape, set(axes))

    elif isinstance(axis, int):
        if axis < 0:
            axis += ndim      
        if axis < 0 or axis >= ndim:
            raise ValueError(f'axis {axis} is out of bounds for array of dimension {ndim}')
        return kdims_manager(shape, {axis})

    else:
        raise TypeError(f'Invalid axis type: {type(axis)}')

def infer_getitem_shape(shape, index):
   import numpy as np
   dummy = np.empty(shape)
   return dummy[*index].shape
