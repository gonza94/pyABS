# core/backend.py

import numpy as np

class _Backend:
    def __init__(self, module):
        self._module = module

    def set(self, module):
        object.__setattr__(self, "_module", module)

    def __getattr__(self, name):
        return getattr(self._module, name)

xp = _Backend(np)

def set_backend(module):
    if isinstance(module, str):
        import importlib
        module = importlib.import_module(module)
    xp.set(module)
