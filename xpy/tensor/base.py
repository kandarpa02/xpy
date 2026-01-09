from typing import Any, Sequence
from .utils import (
    _unbroadcast,
    broadcast_shape,
    name_filler,
    reduced_shape,
    broadcast_to,
    matmul_shape,
    reshape_shape,
    transpose_shape,
    max_min_shape,
    infer_getitem_shape,
)

# from .primitives import xpy

class Tensor:
  def __init__(self, shape=(), parents=(), name=None):
    self.expr_given = name is not None
    self.name = name or name_filler.get_name(base="var")
    self.shape = shape
    self.parents = parents
    self.prim = None
    self.index = None 

  def __str__(self):
      if hasattr(self, 'str'):
          return self.str() #type:ignore
      ret = f"Place<{self.name}>)"
      return ret
    
  def __repr__(self):
    if_name = False
    if self.parents == ():
        if_name = True

    if hasattr(self, 'repr'):
        return self.repr() #type:ignore
    ret = f"Tensor('{self.name}')"
    return ret

  def __hash__(self):
    return id(self)
  
  @staticmethod
  def call(*args, prim: str):
    out = Tensor(parents=args, name=prim)
    out.prim = prim
    return out
