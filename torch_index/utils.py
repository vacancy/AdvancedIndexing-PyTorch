#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/15/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

from functools import reduce
import numpy as np
import torch


class BatchIndicator:
    __slots__ = tuple()


batch = BatchIndicator()


def prod(values, default=1):
    if len(values) == 0:
        return default
    return reduce(lambda x, y: x * y, values)


def is_batched_indexing(arg):
    return isinstance(arg, list) or isinstance(arg, np.ndarray) or torch.is_tensor(arg) or (
        isinstance(arg, slice) and any(map(is_batched_indexing, (arg.start, arg.stop, arg.step)))
    )

def is_batched_int_indexing(arg, batch_dims):
    return is_batched_indexing(arg) and torch.is_tensor(arg) and arg.dim() == batch_dims


def is_batched_list_indexing(arg, batch_dims):
    return is_batched_indexing(arg) and torch.is_tensor(arg) and arg.dim() == batch_dims + 1


def is_batched_bool_indexing(arg, batch_dims):
    return is_batched_indexing(arg) and torch.is_tensor(arg) and arg.dim() == batch_dims and is_bool_indexing(arg)


def is_int_indexing(arg):
    return isinstance(arg, int)


def is_list_indexing(arg):
    return isinstance(arg, list) or (isinstance(arg, np.ndarray) and arg.dtype != np.bool_) or (torch.is_tensor(arg) and arg.dtype != torch.uint8)


def is_bool_indexing(arg):
    return (isinstance(arg, np.ndarray) and arg.dtype == np.bool_) or (torch.is_tensor(arg) and arg.dtype == torch.uint8)


def _expand_pytorch_scalar(x):
    if torch.is_tensor(x):
        if x.dim() == 0:
            return int(x.item())
    return x


def canonize_args(args, dim):
    if not isinstance(args, tuple):
        args = (args, )

    # remove the Ellipsis
    new_args = args
    for i, arg in enumerate(args):
        if arg is Ellipsis:
            new_args = args[:i] + tuple(slice(None, None, None) for _ in range(dim - len(args) + 1)) + args[i+1:]
            break
    args = new_args

    for i, arg in enumerate(args):
        if arg is Ellipsis:
            raise IndexError('Index may contain at most one Ellipsis.')

    args = list(args)
    # if value is a pytorch scalar, cast it as a python int.
    for i, arg in enumerate(args):
        if isinstance(arg, slice):
            args[i] = slice(_expand_pytorch_scalar(arg.start), _expand_pytorch_scalar(arg.stop), _expand_pytorch_scalar(arg.step))
        else:
            args[i] = _expand_pytorch_scalar(arg)

    return tuple(args)


def get_vectorized_dims(args, allow_int):
    if allow_int:
        return tuple(i for i, arg in enumerate(args) if is_list_indexing(arg) or is_bool_indexing(arg) or isinstance(arg, int))
    return tuple(i for i, arg in enumerate(args) if is_list_indexing(arg) or is_bool_indexing(arg))


def get_vectorized_new_dim(args, allow_int=False):
    dims = get_vectorized_dims(args, allow_int)

    if len(dims) == 0:
        return -1

    if not any(is_list_indexing(x) or is_bool_indexing(x) for x in args):
        return -1

    dims = set(dims)
    for i in range(min(dims) + 1, max(dims)):
        if is_int_indexing(args[i]):
            dims.add(i)
    dims = tuple(dims)

    if len(dims) == max(dims) - min(dims) + 1:
        ret = min(dims)
        for i, arg in enumerate(args[:ret]):
            if is_int_indexing(arg):
                ret -= 1
        return ret

    return 0


def infer_batch_dims(args):
    batch_dims = tuple(i for i, arg in enumerate(args) if isinstance(arg, BatchIndicator))
    if not (len(batch_dims) > 0 and max(batch_dims) == len(batch_dims) - 1):
        raise IndexError('Batch dimensions must be the first K dimensions.')
    return len(batch_dims)


def validate_batch_dims(args, batch_dims):
    try:
        for arg in args:
            if is_batched_indexing(arg):
                if isinstance(arg, slice):
                    for x in (arg.start, arg.stop, arg.step):
                        if is_batched_indexing(x):
                            assert torch.is_tensor(x), TypeError('Batched indexing must be torch.tensor.')
                            assert x.dim() == batch_dims
                elif is_bool_indexing(arg):
                    assert torch.is_tensor(arg), TypeError('Batched indexing must be torch.tensor.')
                    assert arg.dim() == batch_dims
                else:
                    assert torch.is_tensor(arg), TypeError('Batched indexing must be torch.tensor.')
                    assert arg.dim() == batch_dims or arg.dim() == batch_dims + 1
    except AssertionError as e:
        if len(e.args) > 0:
            raise e.args[0] from e
        raise IndexError('Inconsistent batched indexing.') from e



def get_batched_vectorized_dims(args, allow_int, batch_dims):
    if allow_int:
        return tuple(i for i, arg in enumerate(args) if (
            is_batched_list_indexing(arg, batch_dims) or is_batched_bool_indexing(arg, batch_dims) or
            is_batched_int_indexing(arg, batch_dims) or is_int_indexing(arg)
        ))

    return tuple(i for i, arg in enumerate(args) if (
        is_batched_list_indexing(arg, batch_dims) or is_batched_bool_indexing(arg, batch_dims)
    ))


def get_batched_vectorized_new_dim(args, batch_dims, allow_int=False):
    dims = get_batched_vectorized_dims(args, allow_int, batch_dims)

    if len(dims) == 0:
        return -1

    if not any(is_batched_list_indexing(x, batch_dims) or is_batched_bool_indexing(x, batch_dims) for x in args):
        return -1

    if len(dims) == max(dims) - min(dims) + 1:
        ret = min(dims)
        for i, arg in enumerate(args[:ret]):
            if is_batched_int_indexing(arg, batch_dims) or is_int_indexing(arg):
                ret -= 1
        return ret
    return batch_dims


def insert_dim(x, src, tgt):
    if src == tgt:
        return x

    rng = list(range(x.dim()))
    del rng[src]
    rng.insert(tgt, src)
    return x.permute(rng)

