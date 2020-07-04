from typing import List
import numpy as np

from xtype import ImmutableTensor
from exp_utils import as_tensor

__all__ = [
    "SelectRows",
    "MatMul",
    "Add",
    "Tanh",
    "Softmax",
]


def SelectRows(rows: List[int], table: ImmutableTensor):
    return as_tensor(table.value[rows, :])


def MatMul(a: ImmutableTensor, b: ImmutableTensor):
    return as_tensor(a.value @ b.value)


def Add(a: ImmutableTensor, b: ImmutableTensor):
    return as_tensor(a.value + b.value)


def Tanh(a: ImmutableTensor):
    return as_tensor(np.tanh(a.value))


def Add(a: ImmutableTensor, b: ImmutableTensor):
    return as_tensor(a.value + b.value)


def Tanh(a: ImmutableTensor):
    return as_tensor(np.tanh(a.value))


def Softmax(a: ImmutableTensor):
    return a
