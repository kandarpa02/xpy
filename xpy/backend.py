from .python_packages import install_with_versions
install_with_versions('2.4.0', None, '13.6.0')

import numpy as _np
_device = "cpu"
_xp = _np       # default

class DeviceError(Exception):
    def __init__(self, *args: object) -> None:
        super().__init__(*args)

def _try_cupy():
    try:
        import cupy as cp
        try:
            cp.cuda.runtime.getDeviceCount()
            return cp
        except Exception:
            pass
        return cp
    except Exception:
        raise DeviceError(f'CUDA requested but drivers not found. ')


def set_device(device: str):
    global _device, _xp

    if device == "cpu":
        _device = "cpu"
        _xp = _np
        return

    if device == "cuda":
        cp = _try_cupy()
        _device = "cuda"
        _xp = cp
        return

    if device == "auto":
        cp = _try_cupy()
        if cp is not None:
            _device = "cuda"
            _xp = cp
        else:
            _device = "cpu"
            _xp = _np
        return

    raise ValueError(f"Unknown device '{device}'")


# # auto detect at import
# set_device("auto")


def xp():
    try:
        import cupy as cp
        try:
            cp.cuda.runtime.getDeviceCount()
            return cp
        except Exception:
            pass
    except ImportError:
        pass

    import numpy as np
    return np


def get_device():
    return _device