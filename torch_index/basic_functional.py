#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : basic_functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/15/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

import torch
from .utils import canonize_args, is_int_indexing, is_list_indexing, is_bool_indexing
from .utils import get_vectorized_new_dim, insert_dim


def reverse_slice(s):
    assert s.step < 0
    max_i = (s.stop + 1 - s.start) // s.step
    return slice(s.start + (max_i) * s.step, s.start + 1, -s.step)


def reverse(x, dim=-1):
    """
    Reverse a tensor along the given dimension. For example, if `dim=0`, it is equivalent to
    the python notation: `x[::-1]`.

    Reference: https://github.com/pytorch/pytorch/issues/229#issuecomment-350041662

    Args:
        x (torch.Tensor): input.
        dim: the dimension to be reversed.

    Returns:
        torch.Tensor: of same shape as `x`, but with the dimension `dim` reversed.

    """

    if x.numel() == 0:
        return x

    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    inds = torch.arange(x.size(1) - 1, -1, -1, dtype=torch.long, device=x.device)
    x = x.view(x.size(0), x.size(1), -1)[:, inds, :]
    return x.view(xsize)


def tindex(value, args):
    args = canonize_args(args, value.dim())

    new_args = list()
    reversed_dims = list()

    j = 0
    for i, arg in enumerate(args):
        if isinstance(arg, slice) and arg.step is not None and arg.step < 0:
            new_args.append(reverse_slice(arg))
            reversed_dims.append(j)
            j += 1
        else:
            new_args.append(arg)
            if is_int_indexing(arg):
                pass
            elif is_list_indexing(arg):
                pass
            else:
                j += 1

    new_value = value[tuple(new_args)]
    new_dim = get_vectorized_new_dim(args)
    for j in reversed_dims:
        if j >= new_dim:
            j += 1
        new_value = reverse(new_value, j)
    return new_value


def findex(value, args):
    args = canonize_args(args, value.dim())

    new_value = tindex(value, args)
    new_dim = get_vectorized_new_dim(args, True)
    if new_dim != -1:
        new_value = insert_dim(new_value, get_vectorized_new_dim(args), new_dim)
    return new_value


def vindex(value, args):
    args = canonize_args(args, value.dim())

    new_value = tindex(value, args)
    new_dim = get_vectorized_new_dim(args)
    if new_dim != -1:
        new_value = insert_dim(new_value, get_vectorized_new_dim(args), 0)
    return new_value


def oindex(value, args):
    args = canonize_args(args, value.dim())

    first = True
    args = list(args)
    rest_indices = list()

    j = 0
    for i, arg in enumerate(args):
        if is_list_indexing(arg) or is_bool_indexing(arg):
            if first:
                first = False
            else:
                rest_indices.append((j, arg))
                args[i] = slice(None, None, None)
            j += 1
        else:
            if not is_int_indexing(arg):
                j += 1
    args = tuple(args)

    new_value = tindex(value, args)
    for j, arg in rest_indices:
        index = (slice(None, None, None), ) * j + (arg, )
        new_value = new_value[index]

    return new_value

