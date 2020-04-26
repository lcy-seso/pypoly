from collections.abc import Sequence
from collections.abc import MutableSequence

import torch
from torch import Tensor

__all__ = [
    'ReadWriteTensorArray',
    'ReadTensorArray',
]


class TensorArray(object):
    pass


class ReadWriteTensorArray(TensorArray, MutableSequence):
    """
    A read-write, list-like class whose elements MUST have a type of torch.Tensor.
    """

    def _type_check(self, input, length, tensor_shape):
        if length is None and tensor_shape is None and input is not None:
            for i in input:
                if not isinstance(i, ReadWriteTensorArray):
                    raise ValueError('Error input.')
            self.length = len(input)
            self.is_leaf = False
            self.T = input

        elif length is not None and tensor_shape is not None and input is None:
            self.is_leaf = True
            self.length = length
            self.T = [[] for _ in range(length)]
        else:
            raise ValueError('Error input.')

    def __init__(self,
                 input=None,
                 length=None,
                 tensor_shape=None,
                 dtype=torch.float32):
        self.tensor_shape = tensor_shape
        self._type_check(input, length, tensor_shape)

    def __str__(self):
        # TODO: for pretty print.
        pass

    def __getitem__(self, i):
        if self.is_leaf:
            return self.T[i][0]
        else:
            return self.T[i]

    def __setitem__(self, index, value):
        if not self.is_leaf:
            raise Exception('Wrong write position.')
        if index >= self.length:
            raise Exception('out-of-bounds access.')
        if self.T[index]:
            raise Exception('Position i is not empty.')
        self.T[index].append(value)

    def __delitem__(self):
        pass

    def __len__(self):
        return self.length

    def insert(self):
        pass

    def gather(self, indices):
        pass

    def scatter(self, indices, value):
        pass


class ReadTensorArray(TensorArray, Sequence):
    """
    A read-only, list-like class whose elements MUST have a type of torch.Tensor.
    """

    def _type_check(self, input):
        if not (isinstance(input, list) or isinstance(input, TensorArray)):
            raise ValueError(
                ('Error input. The Ctor accepts '
                 'a Python built-in list or a TensorArray as input'))

    def __init__(self, input):
        self._type_check(input)
        self.T = input
        super(ReadTensorArray, self).__init__()

    def __getitem__(self, i):
        return self.T[i]

    def __len__(self):
        return len(self.T)

    def gather(self, indices):
        pass
