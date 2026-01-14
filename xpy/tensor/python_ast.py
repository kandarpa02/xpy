import ast
from typing import Sequence, Any, Optional
from .base import Tensor
from ..base import Primitives
from .build_graph import topo_sort, auto_index_leaves, assign_names


def _as_roots(root):
    if isinstance(root, (list, tuple)):
        return tuple(root)
    return (root,)


def build_ast(
    root: Tensor | Sequence[Tensor],
    name: Optional[str] = None,
    inputs: Optional[Sequence[Tensor]] = None
) -> ast.Module:
    """
    Build a Python AST for a computation graph rooted at `root`.
    - `inputs` can be specified explicitly to control function signature.
    """
    name = name or "compiledfunction"
    roots = _as_roots(root)

    # Auto index leaves for temp variables
    auto_index_leaves(roots)
    topo = topo_sort(roots)
    names = assign_names(topo)

    body = []

    # Map leaves in inputs to their function argument names
    if inputs is not None:
        input_names = {t: names[t] for t in inputs}
    else:
        input_names = {t: names[t] for t in topo if t.parents == ()}

    # Generate AST for all intermediate nodes
    for node in topo:
        if node.parents == ():  # skip leaves
            continue

        call = ast.Call(
            func=ast.Attribute(
                value=ast.Name(id="PRIM", ctx=ast.Load()),
                attr=node.prim,
                ctx=ast.Load(),
            ),
            args=[
                # If parent is a function input, use its argument name; else use temp var
                ast.Name(id=input_names.get(p, names[p]), ctx=ast.Load())
                for p in node.parents
            ],
            keywords=[ast.keyword(k, v) for k, v in node.kwds.items()],
        )

        body.append(
            ast.Assign(
                targets=[ast.Name(id=names[node], ctx=ast.Store())],
                value=call,
            )
        )

    # Return statement
    if len(roots) == 1:
        ret = ast.Name(id=names[roots[0]], ctx=ast.Load())
    else:
        ret = ast.Tuple(
            elts=[ast.Name(id=names[r], ctx=ast.Load()) for r in roots],
            ctx=ast.Load(),
        )
    body.append(ast.Return(value=ret))

    # Function arguments
    func_args = [ast.arg(arg=input_names[t]) for t in (inputs or [n for n in topo if n.parents == ()])]

    func_def = ast.FunctionDef(
        name=name,
        args=ast.arguments(
            posonlyargs=[],
            args=func_args,
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=body,
        decorator_list=[],
    )

    return ast.fix_missing_locations(ast.Module(body=[func_def], type_ignores=[]))

