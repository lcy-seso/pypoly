from typing import List
from collections.abc import Sequence
from collections.abc import MutableSequence
import copy

import torch
from torch import Tensor

__all__ = [
    'TensorArray',
    'ReadWriteTensorArray',
    'ReadTensorArray',
    'CellArray',
]


class TensorArray(object):
    """
    A TensorArray is a variable-length list of tensor. Each tensor stored in
    this TensorArray must have a same shape.

    The warped Array class mainly serves for two purposes:

    1. Provide a way to declare complicated memory layout and do necessary
       checks (the main purpose).

        TS Parser (a parser parses a subset of Python syntax) parses text-form
        programs to tokens, if some information does not appear in text-form
        codes, it will not in the parsed tree.

        Python does not have a way to directly declare array size, but use
        dynamic array.

        Array declarations have to be constructed from this wrapped Array.

    2. Make AST clean (not a required requirement):

       Allow us to get array access relation by reading a single AST node
       instead of figuring out the right subtree and then processing the
       subtree to get array access relations.
    """
    pass


class CellArray(TensorArray, Sequence):
    """
    A read-only, list-like class whose elements MUST have a type of
    torch.nn.Module.
    """

    def _type_check(self, input):
        if not (isinstance(input, list) or isinstance(input, TensorArray)):
            raise ValueError(
                ('Error input. The Ctor accepts '
                 'a Python built-in list or a TensorArray as input'))

    def __init__(self, input):
        self._type_check(input)
        self.T = input
        super(CellArray, self).__init__()

    def __getitem__(self, i):
        return self.T[i]

    def __len__(self):
        return len(self.T)


class ReadTensorArray(TensorArray, Sequence):
    """
    A read-only, list-like class whose elements MUST have a type of torch.Tensor.
    """

    def _type_check(self, input):
        array = input
        for i in range(self.dim - 1):
            if not isinstance(input, List):
                raise ValueError(
                    (f'Error input for the {i}th dimension. '
                     f'The Ctor accepts a Python built-in list as the input.'))
            array = array[0]

    def __init__(self, input, array_shape, tensor_shape):
        super(ReadTensorArray, self).__init__()

        self.tensor_shape = tensor_shape if isinstance(
            tensor_shape, List) else list(tensor_shape)
        self.array_shape = array_shape if isinstance(
            array_shape, List) else list(array_shape)
        self.dim = len(array_shape)

        self.length = len(input)

        self._type_check(input)

        self.T = input

    def __str__(self):
        array_shape_str = ', '.join(map(str, self.array_shape))
        tensor_shape_str = ', '.join(map(str, self.tensor_shape))
        return (f'{self.__class__}, array shape = [{array_shape_str}], '
                f'with tensor_shape = [{tensor_shape_str}]')

    def __repr__(self):
        array_shape_str = ', '.join(map(str, self.array_shape))
        tensor_shape_str = ', '.join(map(str, self.tensor_shape))
        return (f'{self.__class__}, array shape = [{array_shape_str}], '
                f'with tensor_shape = [{tensor_shape_str}]')

    def __getitem__(self, i):
        return self.T[i]

    def __len__(self):
        return len(self.T)

    def size(self):
        return self.array_shape

    def gather(self, indices):
        pass


class ReadWriteTensorArrayInternal(TensorArray, MutableSequence):
    def __init__(self, input):
        if not isinstance(input, List):
            raise ValueError(
                ("ReadWriteTensorArrayInternal must be constructed from "
                 "an empty Python list or Python list of "
                 "ReadWriteTensorArrayInternal object."))

        if not input:
            raise ValueError("Error input value.")

        elem = input[0]
        if not ((isinstance(elem, List) and (not elem))
                or isinstance(elem, ReadWriteTensorArrayInternal)):
            raise ValueError(
                ("ReadWriteTensorArrayInternal must be constructed from "
                 "an empty Python list or an ArrayInternal object."))

        self.T = input
        self.is_leaf = not isinstance(input[0], ReadWriteTensorArrayInternal)
        self.length = len(input)

    def __str__(self):
        return f'{self.__class__}, length = {self.length}'

    def __repr__(self):
        return f'{self.__class__}, length = {self.length}'

    def __getitem__(self, index):
        if index >= self.length:
            raise IndexError((f'list index {index} is out of range'
                              f' ({self.length}).'))

        if self.is_leaf:
            if not self.T[index]:
                raise IndexError((f'Access uninitalized array element '
                                  f'at position {index}.'))

            return self.T[index][0]
        else:
            return self.T[index]

    def __setitem__(self, index, value):
        if not self.is_leaf:
            raise Exception(f'Wrong write position {index}.')
        if index >= self.length:
            raise IndexError((f'list index {index} is out of range'
                              f' ({self.length}).'))
        if self.T[index]:
            raise Exception(f'Position {index} cannot be written twice.')
        self.T[index].append(value)

    def __delitem__(self):
        pass

    def __len__(self):
        return self.length

    def size(self):
        return self.array_shape

    def insert(self):
        pass


class ReadWriteTensorArray(TensorArray, MutableSequence):
    """
    A read-write, list-like class whose elements MUST have a type of
    torch.Tensor.
    """

    def __init__(self,
                 array_shape,
                 tensor_shape,
                 dtype=torch.float32,
                 **kwargs):
        super(ReadWriteTensorArray, self).__init__()

        self.tensor_shape = tensor_shape if isinstance(
            tensor_shape, List) else list(tensor_shape)
        self.array_shape = array_shape if isinstance(
            array_shape, List) else list(array_shape)
        self.dim = len(array_shape)

        self.T = self._create_array()

    def _create_array(self):
        # the innermost dimension.
        list_inner_dim = ReadWriteTensorArrayInternal(
            [[] for _ in range(self.array_shape[-1])])

        list_current_dim = []
        for dim_size in reversed(self.array_shape[:-1]):
            list_current_dim = []
            for i in range(dim_size):
                list_current_dim.append(copy.deepcopy(list_inner_dim))
            list_inner_dim = ReadWriteTensorArrayInternal(list_current_dim)
        return list_current_dim

    def _array_shape_str(self):
        array_shape = []
        array_dim = self.T
        for i in range(self.dim):
            array_shape.append(len(array_dim))
            try:
                array_dim = array_dim[0]
            except IndexError:
                pass

        return ', '.join(map(str, array_shape))

    def __str__(self):
        array_shape_str = self._array_shape_str()
        tensor_shape_str = ', '.join(map(str, self.tensor_shape))
        return (f'{self.__class__}, array shape = [{array_shape_str}], '
                f'with tensor_shape = [{tensor_shape_str}]')

    def __repr__(self):
        array_shape_str = self._array_shape_str()
        tensor_shape_str = ', '.join(map(str, self.tensor_shape))
        return (f'{self.__class__}, array shape = [{array_shape_str}], '
                f'with tensor_shape = [{tensor_shape_str}]')

    def __getitem__(self, i):
        return self.T[i]

    def __setitem__(self, index, value):
        if not isinstance(value, ReadWriteTensorArrayInternal):
            raise ValueError(('The element of ReadWriteTensorArray must be '
                              'a ReadWriteTensorArrayInternal object.'))
        if index >= self.array_shape[0]:
            raise IndexError((f'list index {index} is out of range'
                              f' ({self.length}).'))
        self.T[index] = value

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
