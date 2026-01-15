from .base import primitive

dmap = {'cupy':'cuda', 'numpy':'cpu'}

def shift_dvice_(data, device:str):
  if not device.lower() in {'cpu', 'cuda'}: 
    raise TypeError(f"Unknown device '{device}'. Only 'cpu' and 'cuda' is supported")
  from .backend import xp
  lib = xp()
  import numpy as np
  if device == 'cuda':
    if lib.__name__ == 'cupy':
      return lib.array(data)
    else:
      raise ValueError(f"'cuda' device is not available, try 'cpu'. ")
  
  return np.array(data)