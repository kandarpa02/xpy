# from .python_ast import build_ast
# from .base import Tensor
# from ..base import NpPrimitives, CpPrimitives
# from typing import Callable, Sequence, Optional


# def forward(
#     root: Tensor | Sequence[Tensor],
#     name: Optional[str] = None,
#     inputs: Optional[Sequence[Tensor]] = None
# ):
#     """
#     Compile a computation graph into a Python function.
#     - `inputs` explicitly defines the function arguments.
#     """
#     module = build_ast(root, name=name, inputs=inputs)
#     code = compile(module, filename="compiledfunction", mode="exec")
#     namespace = {"PRIM": Primitives()}
#     exec(code, namespace)
#     return namespace[name or "compiledfunction"]
