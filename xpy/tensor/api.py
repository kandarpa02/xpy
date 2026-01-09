from .python_ast import build_ast
from .base import Tensor
from ..base import Primitives
from typing import Callable, Sequence

def forward(root:Sequence[Tensor]|Tensor, name:str|None=None) -> Callable:

  module = build_ast(root, name)
  code = compile(module, mode='exec', filename='compiledfunction')
  namespace = {'PRIM':Primitives()}
  exec(code, namespace)

  return namespace[name if name is not None else "compiledfunction"]