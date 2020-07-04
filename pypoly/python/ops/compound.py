import pdb
import numpy as np

import torch
from torch.autograd import Function

from . import meta
from ..array import ImmutableArray

__all__ = [
    'MatMul',
]


class MatMulFunction(Function):
    """Matrix A with a shape [M, N] multiply matrix B with a shape[N, K]."""

    @staticmethod
    def forward(ctx, a: ImmutableArray, b: ImmutableArray) -> ImmutableArray:
        a_row = a.shape[0]
        a_col = a.shape[1]

        b_row = b.shape[0]
        b_col = b.shape[1]

        output = np.zeros((a_row, b_col), dtype=np.float32)
        for i in range(a_row):
            for j in range(b_col):
                row = meta.SelectRows(i, a)
                col = meta.SelectColums(j, b)
                output[i, j] = meta.DotProduct(row, col)
        return torch.as_tensor(output, dtype=torch.float)

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError()


def MatMul(a: ImmutableArray, b: ImmutableArray):
    return MatMulFunction.apply(a, b)
