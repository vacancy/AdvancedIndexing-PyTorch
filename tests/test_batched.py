#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_basic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

import torch
import numpy as np
from torch_index import batch
from torch_index import btindex, bfindex, bvindex, boindex
from torch_index import tindex, findex, vindex, oindex


class _MakeIndex(object):
    def __getitem__(self, item):
        return item

_mi = _MakeIndex()


def test_batched_index():
    a = torch.rand(2, 3, 6, 7, 8)

    blist = torch.tensor([
        [[0, 1], [0, 3], [0, 5]],
        [[1, 2], [1, 3], [1, 5]]
    ])
    assert blist.shape == (2, 3, 2)
    bint0 = blist[:, :, 0]
    bint1 = blist[:, :, 1]
    assert bint0.shape == (2, 3)
    assert bint1.shape == (2, 3)

    blist_length = torch.tensor([
        [1, 2, 2],
        [2, 1, 2]
    ])

    for func, batched_func in zip([tindex, findex, vindex, oindex], [btindex, bfindex, bvindex, boindex]):
        yield check_indexing, func, batched_func, a, _mi[batch, batch, bint0, bint1]
        yield check_indexing, func, batched_func, a, _mi[batch, batch, bint0, bint1, blist]
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, bint1, blist]
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, 3:5, blist]
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, bint0:bint1, blist]
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, 5:bint0:-1, blist]

        yield check_indexing, func, batched_func, a, _mi[batch, batch, bint0, bint1, blist], blist_length
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, bint1, blist], blist_length
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, 3:5, blist], blist_length
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, bint0:bint1, blist], blist_length
        yield check_indexing, func, batched_func, a, _mi[batch, batch, blist, 5:bint0:-1, blist], blist_length



def extract_peritem_args(args, i, j, indices_length):
    def get(x):
        if x is None:
            return x
        if isinstance(x, int):
            return x

        if torch.is_tensor(x):
            y = x[i, j]
            if y.dim() == 1 and indices_length is not None:
                y = y[:indices_length[i, j]]
            return y
        return x

    def gen():
        for x in args[2:]:
            if isinstance(x, slice):
                yield slice(get(x.start), get(x.stop), get(x.step))
            else:
                yield get(x)
    return tuple(gen())


def expanded_batched_index(func, value, args, indices_length):
    assert args[0] == batch and args[1] == batch

    output = [[None for j in range(value.shape[1])] for i in range(value.shape[0])]
    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            this_args = extract_peritem_args(args, i, j, indices_length)
            output[i][j] = func(value[i, j])[this_args]

    return output


def check_indexing(func, batched_func, value, args, indices_length=None, debug_string=None):
    if debug_string is not None:
        print(debug_string)

    output_0 = expanded_batched_index(func, value, args, indices_length)
    output, shape = batched_func(value, indices_length)[args]

    for i in range(value.shape[0]):
        for j in range(value.shape[1]):
            print(i, j, extract_peritem_args(args, i, j, indices_length))
            assert tuple(shape[i, j].tolist()) == output_0[i][j].shape, (tuple(shape[i, j].tolist()), output_0[i][j].shape)

            output_index = (i, j) + tuple(slice(None, shape[i, j, x].item(), None) for x in range(shape.size(-1)))
            assert torch.allclose(output[output_index], output_0[i][j])

