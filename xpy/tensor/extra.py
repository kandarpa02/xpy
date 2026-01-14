def broadcast_backward(grad, x):
  from ..tensor.base import Tensor

  x_shape = x.shape
  while len(grad.shape) > len(x_shape):
    _grad = Tensor.call(grad, prim='sum', params={'axis':0})

  for i, (sx, sg) in enumerate(zip(x_shape, grad.shape)):
    if sx == 1 and sg != 1:
      _grad = Tensor.call(grad, prim='sum', params={'axis':i, 'keepdims':True})

  return _grad
