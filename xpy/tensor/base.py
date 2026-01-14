from typing import Any, Sequence, Callable
from .utils import name_filler
from ..backend  import xp
lib = xp()

import ast

def literal_to_ast(v):
  if isinstance(v, (int, float, str, bool, lib.ndarray)) or v is None:
    return ast.Constant(value=v)
  elif isinstance(v, (list, tuple)):
    return ast.List(
      elts=[literal_to_ast(x) for x in v],
      ctx=ast.Load()
    )
  elif isinstance(v, dict):
    return ast.Dict(
      keys=[literal_to_ast(k) for k in v.keys()],
      values=[literal_to_ast(val) for val in v.values()],
    )
  else:
    raise TypeError(f"Unsupported literal type in AST: {type(v)}")


class Tensor:
  def __init__(self, shape=(), parents=(), name=None, params:dict={}):
    self.expr_given = name is not None
    self.name = name or name_filler.get_name(base="var")
    self.shape = shape
    self.parents = parents
    self.prim = None
    self.index = None 
    self.kwds = {k:literal_to_ast(v) for k, v in params.items()}

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
  def call(*args, prim: str, params:dict={}):
    out = Tensor(parents=args, name=prim, params=params)
    out.prim = prim
    return out
  
  @staticmethod
  def constant(value:Any):
    return literal_to_ast(value)
  

class GFunc:
  def __init__(self) -> None:
    self.parents = None

  def save(self, *args):
    self.parents = list(args)

  def saved_tensor(self):
    return self.parents[0] if len(self.parents)==1 else tuple(self.parents)

  def forward(self, *args):
      raise NotImplementedError
  
  def backward(self, grad):
    raise NotImplementedError
  
  @classmethod
  def apply(cls, *args):
    return cls().forward(*args)
  