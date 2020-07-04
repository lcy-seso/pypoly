import pdb

from typing import List
import numpy as np
from ..array import ImmutableTensor, ImmutableArray

import torch
from torch.autograd import Function

__all__ = [
    'SelectRows',
    'SelectColums',
    'DotProduct',
]


class SelectColumsFunction(Function):
    @staticmethod
    def forward(ctx, cols: List[int],
                table: ImmutableTensor) -> ImmutableTensor:
        return table[:, cols]

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


def SelectColums(cols: List[int], table: ImmutableTensor) -> ImmutableTensor:
    return SelectColumsFunction.apply(cols, table)


class SelectRowsFunction(Function):
    @staticmethod
    def forward(ctx, rows: List[int], table: ImmutableTensor):
        return table[rows, :]

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


def SelectRows(rows: ImmutableArray,
               table: ImmutableTensor) -> ImmutableTensor:
    return SelectRowsFunction.apply(rows, table)


class DotProductFunction(Function):
    @staticmethod
    def forward(ctx, a: ImmutableTensor,
                b: ImmutableTensor) -> ImmutableTensor:
        a = a.detach()
        b = b.detach()
        result = np.dot(a, b)
        return torch.as_tensor(result, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


def DotProduct(a: ImmutableTensor, b: ImmutableTensor) -> ImmutableTensor:
    return DotProductFunction.apply(a, b)
