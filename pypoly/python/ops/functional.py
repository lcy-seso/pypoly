import pdb

from typing import List
import numpy as np

import torch
from torch.autograd import Function

from ..array import ImmutableArray

__all__ = [
    "iterator",
    "foreach",
]


class Iterator(object):
    pass


class Expression(object):
    pass


def foreach(input: ImmutableArray, body: callable) -> ImmutableArray:
    output = []
    for i, item in enumerate(input):
        out = body(i, item)
        output.append(out)
    return output
