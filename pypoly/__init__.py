from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import inspect
import types
import collections
import warnings
import torch
from typing import List

from .python import array_pb2
from .python.cells import *
from .python import *

__all__ = [
    'ScopParser',
] + python.__all__

#FIXME(Ying) for debug only, Needs a standarded way to distribute the package
# and import bindings.
import _parser

_parser.init_glog(sys.argv[0])


class FnStub(
        collections.namedtuple('FnStub',
                               ('source', 'filename', 'file_lineno'))):
    pass


class ScopParser(object):
    def __init__(self, nn_module):
        super(ScopParser, self).__init__()
        self.nn_module = nn_module
        self.fn_stubs = self._get_fn_stubs(nn_module)

        # TODO(Ying): Add recursively parsing.
        source, filename, file_lineno = self.fn_stubs[0]
        self.asts = [_parser.get_torch_ast(source, filename, file_lineno)]
        self.var_contexts = []

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass

    def _get_fn_stubs(self, nn_module):
        r"""Given a subclass of torch.nn.Moudule, recursively get all functions
        to be analyzed.

        Args:
            nn_module: a subclass of torch.nn.Module.
        """

        fn_stubs = []
        if isinstance(nn_module, torch.nn.Module):
            if not hasattr(nn_module, 'forward'):
                raise Exception('forward implementation is not found.')
            method = getattr(nn_module, 'forward', None)

            filename = inspect.getsourcefile(method)
            sourcelines, file_lineno = inspect.getsourcelines(method)
            source = ''.join(sourcelines)
            fn_stubs.append(
                FnStub(
                    source=source, filename=filename, file_lineno=file_lineno))
        else:
            raise NotImplementedError(
                'parse codes other than nn.Module is not implemented yet.')
        return fn_stubs

    def optimize(self, scop):
        # TODO(Ying): Not implemented yet. Make optimized codes a callable
        # object and return it.
        return self.nn_module

    def print_ast(self):
        print(self.asts[0])

    def parse(self):
        contexts = array_pb2.ContextDesc()
        contexts.vars.extend(self.var_contexts)

        parsed = _parser.parse_scop(self.asts[0], contexts.SerializeToString())
        return self.optimize(parsed)

    def add_int(self, var, lower=None, upper=None):
        """Add value range of a variable.

        Currently we consider the following type of var:
            1. an integral value:
            2. a nested list:

        Args:
            min_val: int, Note the defualt value `-sys.maxsize - 1` is the
                     minimal value representable by a signed word.
            max_val: int, Note the defualt value `-sys.maxsize - 1` is the
                     maximal value representable by a signed word.
        """

        # Given a Python object, inspect its string-literal name.
        # Note: the implementation is not stable. It looks for variable name
        # from next outer frame object (this frame’s caller).
        var_name = None
        caller_locals = inspect.currentframe().f_back.f_locals
        for name in caller_locals:
            if caller_locals[name] is var:
                var_name = name
                break
        if var_name is None:
            raise ValueError("Fail to inspect variable name.")

        if not isinstance(var, int):
            raise ValueError("unsppported type. `var` must be an integer.")

        # The program parameter is an integer, for example, batch size,
        # sequence length.
        if lower is None or upper is None:
            warnings.warn(
                (f'Value bound of {var_name} is not given, '
                 f'so the default value bound is used '
                 f'with lower_bound = {lower}, upper_bound = {upper}.'),
                UserWarning)

        lower = -sys.maxsize - 1 if lower is None else lower
        upper = sys.maxsize if upper is None else upper

        context_var = array_pb2.ContextVar()
        context_var.name = var_name
        context_var.type = array_pb2.ContextVarType.INT32
        context_var.lower_bound.extend([lower])
        context_var.upper_bound.extend([upper])

        context_var.elem_desc.elem_type = 'int'
        context_var.elem_desc.shape.extend([1])

        self.var_contexts.append(context_var)

    def add_array(self, var, min_shape=None, max_shape=None):
        def _get_valid_shape(shape1, shape2, cmp_func):
            assert len(shape1) == len(shape2)
            return [cmp_func(s1, s2) for s1, s2 in zip(shape1, shape2)]

        # Given a Python object, inspect its string-literal name.
        # Note: the implementation is not stable. It looks for variable name
        # from next outer frame object (this frame’s caller).
        var_name = None
        caller_locals = inspect.currentframe().f_back.f_locals
        for name in caller_locals:
            if caller_locals[name] is var:
                var_name = name
                break
        if var_name is None:
            raise ValueError("Fail to inspect variable name.")

        if isinstance(var, List):
            if not var:
                raise ValueError("empty list is an invalid context variable.")
            for idx, value in enumerate(var):
                if not isinstance(value, int):
                    raise ValueError(
                        (f'The {idx}th element is not an integer. '
                         f'When `var` is a Python list, '
                         f'all its elements must be integers.'))

            min_shape = [1] if min_shape is None else _get_valid_shape(
                min_shape, var.shape, min)
            max_shape = ([len(var)] if max_shape is None else _get_valid_shape(
                max_shape, var.shape))

            context_var = array_pb2.ContextVar()
            context_var.name = var_name
            context_var.type = array_pb2.ContextVarType.INT32_ARRAY
            context_var.lower_bound.extend(min_shape)
            context_var.upper_bound.extend(max_shape)

            context_var.elem_desc.elem_type = 'int'
            context_var.elem_desc.shape.extend([1])

            self.var_contexts.append(context_var)
        elif isinstance(var, TensorArray):
            min_shape = [1 for _ in range(var.dim)
                         ] if min_shape is None else min_shape
            max_shape = var.array_shape if max_shape is None else max_shape
            array_type = ('read_tensor_array' if isinstance(
                var, ReadTensorArray) else 'readwrite_tensor_array')

            context_var = array_pb2.ContextVar()
            context_var.name = var_name
            context_var.type = array_pb2.ContextVarType.TENSOR_ARRAY
            context_var.lower_bound.extend(min_shape)
            context_var.upper_bound.extend(max_shape)

            context_var.elem_desc.elem_type = 'tensor'
            context_var.elem_desc.shape.extend(var.tensor_shape)

            self.var_contexts.append(context_var)
        else:
            raise ValueError("unsppported type.")

    def print_context(self):
        contexts = array_pb2.ContextDesc()
        contexts.vars.extend(self.var_contexts)
        print(contexts)
