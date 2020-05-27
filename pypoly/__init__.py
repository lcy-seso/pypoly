from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import inspect
import torch

from .python.cells import *
from .python import *

__all__ = [
    'scop',
] + python.__all__

#FIXME(Ying) for debug only, Use a standarded way to import bindings.
import _parser

_parser.init_glog(sys.argv[0])


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

        parsed = _parser.parse_scop(source, filename, file_lineno)

    else:
        raise NotImplementedError(
            'parse codes other than nn.Module is not implemented yet.')
