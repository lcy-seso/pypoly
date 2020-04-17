import __future__

import torch

from torch.jit._recursive import concrete_type_store

from .cells import *
from .tensor_array import TensorArray


def scop(nn_module):
    r"""parse static control parts.
    """
    if isinstance(nn_module, torch.nn.Module):
        if not hasattr(nn_module, 'forward'):
            raise Exception('forward implementation is not found.')
        method = getattr(nn_module, 'forward', None)

        concrete_type = concrete_type_store.get_or_create_concrete_type(
            nn_module)
        methods = torch.jit._recursive.infer_methods_to_compile(nn_module)
        for method in methods:
            print(method.def_)

    else:
        raise NotImplementedError(
            'parse codes other than nn.Module is not implemented yet.')
