import pdb

from typing import List, Tuple

from collections.abc import Sequence
from collections.abc import MutableSequence

from abc import ABC, abstractmethod

__all__ = [
    "Float",
    "ImmutableTensor",
    "MutableTensor",
    "MutableArray",
    "ImmutableArray",
]


class Iterator(int):
    pass


class Float(float):
    def __init__(self, *args, **kwargs):
        pass


class Array(ABC):
    # TODO: inherit from metaclass.
    pass


class Tensor(ABC):
    # TODO: inherit from metaclass.
    pass


class ImmutableTensor(Tensor):
    def __init__(self, value):
        super(ImmutbleTensor, self).__init__()
        self.shape = value.shape
        self.dtype = dtype
        self.value = value

    def __str__(self):
        return 'ImmutableTensor: ' + str(self.value)

    __repr__ = __str__


class MutableTensor(Tensor):
    def __init__(self, shape, dtype, value=None):
        super(MutableTensor, self).__init__()

        self.shape = shape
        self.dtype = dtype
        self.value = value

        self.is_leaf = True

    def __str__(self):
        return 'MutableTensor: ' + str(self.value)

    __repr__ = __str__


class Array(ABC):
    # TODO: inherit from metaclass.

    def __init__(self, elements: List):
        self.T = elements

    def foreach(self, func: callable):
        output = []
        for i, input in enumerate(self.T):
            out = func(i, input)
            output.append(out)
        return ImmutableArray(output)

    def scan(self, func: callable, init, **kwargs):
        output = []
        for i, input in enumerate(self.T):
            if i == 0:
                out = func(i, init, input, **kwargs)
            else:
                out = func(i, output[i - 1][0], input, **kwargs)
            output.append(out)
        return ImmutableArray(output)

    def fold(self, func: callable, init, **kwargs):
        # TODO: Fix that the variable name and implementation details are
        # hardcoded for demonstration.
        h_prev = kwargs['h_prev']

        output = []
        for i, input in enumerate(self.T):
            if i == 0:
                x = init
            else:
                x = output[-1]

            out = func(x, h_prev, *input)
            output.append(out)
        return ImmutableArray([output[-1]])

    def reduce(self, func: callable):
        x = self.T[0][0]
        for i, input in enumerate(self.T[1:]):
            x = func(x, input[0])
        return ImmutableArray([x])


class ImmutableArray(Array, Sequence):
    def __init__(self, elements: List):
        """Constructed from nested Python array."""

        if isinstance(elements, List):
            if not elements:
                raise RuntimeError("Empty list is not allowe.")

            first_element = elements[0]
            if not isinstance(first_element, ImmutableArray):
                if isinstance(first_element, List):
                    # FIXME(Ying): Construct a correct type inheritance
                    # relationships and fix the check.
                    raise RuntimeError("Wrong initialize value.")
            else:
                first_element = first_element[0]
        elif isinstance(elements, ImmutableArray):
            pass
        else:
            raise RuntimeError("Wrong initialize value.")

        self.shape = len(elements)

        # FIXME(Ying): a consistent way to describe type.
        self.dtype = elements[0].__class__.__name__

        self.T = elements

        super(ImmutableArray).__init__()

    def __str__(self):
        return (f'{self.__class__.__name__}, shape = {self.shape}, '
                f'dtype = {self.dtype}')

    __repr__ = __str__

    def __getitem__(self, i):
        return self.T[i]

    def __len__(self):
        return len(self.T)

    def last(self):
        return self.T[-1]


class MutableArray(Array, MutableSequence):
    def __init__(self, shape, dtype):
        super(MutableArray).__init__()

        self.shape = shape
        self.dtype = dtype

    def __str__(self):
        return f'{self.__class__}, length = {self.shape}'

    def __repr__(self):
        return f'{self.__class__}, length = {self.shape}'

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
