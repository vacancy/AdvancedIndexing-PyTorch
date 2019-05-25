#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_basic.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

import numpy as np
import torch
import torch_index


class _MakeIndex(object):
    def __getitem__(self, item):
        return item

_mi = _MakeIndex()


def check_indexing_numpy(func, tensor, index):
    numpy_index = list()
    for x in index:
        if torch.is_tensor(x):
            numpy_index.append(x.numpy().astype(np.bool_) if x.dtype == torch.uint8 else x.numpy())
        else:
            numpy_index.append(x)
    numpy_index = tuple(numpy_index)
    a, b = func(tensor)[index].numpy(), tensor.numpy()[numpy_index]
    assert a.shape == b.shape, 'Inconsistent shape: a={}, b={}, index={}.'.format(a.shape, b.shape, index)
    assert np.allclose(a, b), 'Inconsistent value: a={}, b={}, index={}.'.format(a.shape, b.shape, index)


def check_indexing_gt(func, tensor, index, gt):
    a, b = func(tensor)[index].shape, gt
    assert a == b, 'Inconsistent shape: a={}, b={}, index={}.'.format(a, b, index)


def test_basic_findex():
    a = torch.rand(5, 6, 7, 8)
    bindx = torch.zeros((7, 8), dtype=torch.uint8)
    bindx[0, 0] = 1

    yield check_indexing_numpy, torch_index.findex, a, _mi[[0], ...]
    yield check_indexing_numpy, torch_index.findex, a, _mi[3:4, [0], ...]
    yield check_indexing_numpy, torch_index.findex, a, _mi[4:2:-1, [0], ...]
    yield check_indexing_numpy, torch_index.findex, a, _mi[[0, 1], 4:2:-1, ..., [0, 1]]
    yield check_indexing_numpy, torch_index.findex, a, _mi[4:2:-1, [0, 1], ..., [0, 1]]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, [0], [0], :]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, [0], :, [0]]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, [0], 0, :]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, [0], :, 0]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, [0, 1], 0, [0, 1]]
    yield check_indexing_numpy, torch_index.findex, a, _mi[:, 0, bindx]
    yield check_indexing_numpy, torch_index.findex, a, _mi[0, :, bindx]
    yield check_indexing_numpy, torch_index.findex, a, _mi[[0], :, bindx]

    yield check_indexing_gt, torch_index.vindex, a, _mi[:, [0, 1], :, [0, 1]], (2, 5, 7)
    yield check_indexing_gt, torch_index.vindex, a, _mi[:, [0, 1], [0, 1]], (2, 5, 8)
    yield check_indexing_gt, torch_index.vindex, a, _mi[4:2:-1, [0, 1], [0, 1]], (2, 2, 8)
    yield check_indexing_gt, torch_index.oindex, a, _mi[[0, 1], [0, 1], bindx], (2, 2, 1)

    yield check_indexing_gt, torch_index.oindex, a, _mi[:, [0, 1], :, [0, 1]], (5, 2, 7, 2)
    yield check_indexing_gt, torch_index.oindex, a, _mi[:, [0, 1], [0, 1]], (5, 2, 2, 8)
    yield check_indexing_gt, torch_index.oindex, a, _mi[4:2:-1, [0, 1], [0, 1]], (2, 2, 2, 8)
    yield check_indexing_gt, torch_index.oindex, a, _mi[[0, 1], [0, 1], bindx], (2, 2, 1)

