#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/15/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
import torch


def _expand_pytorch_scalar(x):
    if torch.is_tensor(x):
        if x.dim() == 0:
            return int(x.item())
    return x


def _clamp(x, min_value, max_value):
    return min(max(x, min_value), max_value)


def is_list_indexing(arg):
    return isinstance(arg, list) or (isinstance(arg, np.ndarray) and arg.dtype != np.bool_) or (torch.is_tensor(arg) and arg.dtype != torch.uint8)


def is_bool_indexing(arg):
    return (isinstance(arg, np.ndarray) and arg.dtype == np.bool_) or (torch.is_tensor(arg) and arg.dtype == torch.uint8)


def expand_index(args, dim):
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


def reverse_slice(s):
    max_i = (s.stop + 1 - s.start) // s.step
    return slice(s.start + (max_i) * s.step, s.start + 1, -s.step)


def get_vectorized_dims(args, allow_int):
    if allow_int:
        return [i for i, arg in enumerate(args) if is_list_indexing(arg) or is_bool_indexing(arg) or isinstance(arg, int)]
    return [i for i, arg in enumerate(args) if is_list_indexing(arg) or is_bool_indexing(arg)]


def get_vectorized_new_dim(args, allow_int=False):
    dims = get_vectorized_dims(args, allow_int)

    if len(dims) == 0:
        return -1

    if not any(is_list_indexing(x) or is_bool_indexing(x) for x in args):
        return -1

    if len(dims) == max(dims) - min(dims) + 1:
        ret = min(dims)
        for i, arg in enumerate(args[:ret]):
            if isinstance(arg, int):
                ret -= 1
        return ret
    return 0


def insert_dim(x, src, tgt):
    if src == tgt:
        return x

    rng = list(range(x.dim()))
    del rng[src]
    rng.insert(tgt, src)
    return x.permute(rng)


def reverse(x, dim=-1):
    """
    Reverse a tensor along the given dimension. For example, if `dim=0`, it is equivalent to
    the python notation: `x[::-1]`.

    Args:
        x (torch.Tensor): input.
        dim: the dimension to be reversed.

    Returns:
        torch.Tensor: of same shape as `x`, but with the dimension `dim` reversed.

    """
    # https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662

    if x.numel() == 0:
        return x

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    inds = torch.arange(x.size(1) - 1, -1, -1, dtype=torch.long, device=x.device)
    x = x.view(x.size(0), x.size(1), -1)[:, inds, :]
    return x.view(xsize)


class IndexWrapper(object):
    __index_func__ = None

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, args):
        return type(self).__index_func__(self.tensor, args)

