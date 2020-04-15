import __future__

import torch

from .cells import *
from .tensor_array import TensorArray


def scop(obj):
    r"""parse static control parts.
    """
    if isinstance(obj, torch.nn.Module):
        if not hasattr(obj, 'forward'):
            raise Exception('forward implementation is not found.')
        method = getattr(obj, 'forward', None)

        # TODO(Ying) recursive parsing is required.
        return torch.jit.get_jit_def(method)
    else:
        raise NotImplementedError(
            'parse codes other than nn.Module is not implemented yet.')
