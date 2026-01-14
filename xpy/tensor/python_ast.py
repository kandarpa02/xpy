from ast import Name, FunctionDef, Return, Call, Module, arg, arguments
from typing import Callable
from .build_graph import topo_sort, auto_index_leaves, assign_names
from ..tensor.base import Tensor

import ast

def _as_roots(root):
    if isinstance(root, (list, tuple)):
        return tuple(root)
    return (root,)

def build_ast(root, name=None, inputs: list[Tensor]|None=None):
    name = name if name is not None else 'compiledfunction'
    roots = _as_roots(root)

    auto_index_leaves(roots)
    topo = topo_sort(roots)
    names = assign_names(topo)

    body = []

    # Assign intermediate nodes (same as before)
    for node in topo:
        if node.parents == ():
            continue
        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="PRIM", ctx=ast.Load()),
                attr=node.prim,
                ctx=ast.Load(),
            ),
            args=[ast.Name(id=names[p], ctx=ast.Load()) for p in node.parents],
            keywords=[ast.keyword(k, v) for k, v in node.kwds.items()],
        )
        body.append(ast.Assign(targets=[ast.Name(id=names[node], ctx=ast.Store())], value=call))

    # Return
    if len(roots) == 1:
        ret = ast.Name(id=names[roots[0]], ctx=ast.Load())
    else:
        ret = ast.Tuple(
            elts=[ast.Name(id=names[r], ctx=ast.Load()) for r in roots],
            ctx=ast.Load(),
        )
    body.append(ast.Return(value=ret))

    # --- Function args ---
    if inputs is not None:
        # Explicit inputs override auto leaf ordering
        args = [ast.arg(arg=names[t]) for t in inputs]
    else:
        # Default: all leaf nodes
        args = [ast.arg(arg=f"x{n.index}") for n in topo if n.parents == ()]

    func = ast.FunctionDef( #type:ignore
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=args,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )

    return ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
