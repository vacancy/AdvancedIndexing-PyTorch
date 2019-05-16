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
from .utils import expand_index, is_list_indexing, is_bool_indexing, reverse_slice, reverse
from .utils import get_vectorized_new_dim, insert_dim


def tindex(value, args):
    args = expand_index(args, value.dim())

    new_args = list()
    reversed_dims = list()

    j = 0
    for i, arg in enumerate(args):
        if isinstance(arg, slice) and arg.step is not None and arg.step < 0:
            new_args.append(reverse_slice(arg))
            reversed_dims.append(j)
        else:
            new_args.append(arg)
            if isinstance(arg, int):
                pass
            else:
                j += 1

    new_value = value[new_args]
    for j in reversed_dims:
        new_value = reverse(new_value, j)
    return new_value


def findex(value, args):
    args = expand_index(args, value.dim())

    new_value = tindex(value, args)
    if get_vectorized_new_dim(args, True) == 0:
        return insert_dim(new_value, get_vectorized_new_dim(args), get_vectorized_new_dim(args, True))
    return new_value


def vindex(value, args):
    args = expand_index(args, value.dim())

    new_value = tindex(value, args)
    return insert_dim(new_value, get_vectorized_new_dim(args), 0)


def oindex(value, args):
    args = expand_index(args, value.dim())

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
            if not isinstance(arg, int):
                j += 1
    args = tuple(args)

    new_value = tindex(value, args)
    for j, arg in rest_indices:
        index = (slice(None, None, None), ) * j + (arg, )
        new_value = new_value[index]

    return new_value

