import __future__
import inspect

import torch

from torch.jit._recursive import concrete_type_store

from .cells import *
from .tensor_array import TensorArray

#FIXME(Ying) for debug only, Use a standarded way to import bindings.
import _parser


def scop(nn_module):
    r"""parse static control parts.
    """
    if isinstance(nn_module, torch.nn.Module):
        if not hasattr(nn_module, 'forward'):
            raise Exception('forward implementation is not found.')
        method = getattr(nn_module, 'forward', None)

        filename = inspect.getsourcefile(method)
        sourcelines, file_lineno = inspect.getsourcelines(method)
        source = ''.join(sourcelines)

        parsed = _parser.parse_scop(source)

    else:
        raise NotImplementedError(
            'parse codes other than nn.Module is not implemented yet.')
