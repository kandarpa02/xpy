from .base import primitive

dmap = {'cupy':'cuda', 'numpy':'cpu'}

def shift_device_(data, device: str):
  device = device.lower()
  if device not in {"cpu", "cuda"}:
    raise TypeError(
      f"Unknown device '{device}'. Only 'cpu' and 'cuda' are supported"
    )

  from .backend import xp
  lib = xp()

  import numpy as np

  if device == "cuda":
    if lib.__name__ != "cupy":
        raise ValueError("'cuda' device is not available, try 'cpu'.")

    if isinstance(data, np.ndarray):
      return lib.asarray(data)

    # already cupy
    if isinstance(data, lib.ndarray):
      return data

    return lib.asarray(data)

  if lib.__name__ == "cupy":
    if isinstance(data, lib.ndarray):
      return data.get()   # 🔥 required

  return np.asarray(data)
