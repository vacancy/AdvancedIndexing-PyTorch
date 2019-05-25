#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : torch_utils.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/25/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

"""
Utility functions from the `jactorch.functional` package: https://github.com/vacancy/Jacinle
"""

import collections
import torch


def concat_shape(*shapes):
    output = []
    for s in shapes:
        if isinstance(s, collections.Sequence):
            output.extend(s)
        else:
            output.append(int(s))
    return tuple(output)


def broadcast(tensor, dim, size):
    if dim < 0:
        dim += tensor.dim()
    assert tensor.size(dim) == 1
    shape = tensor.size()
    return tensor.expand(concat_shape(shape[:dim], size, shape[dim+1:]))


def add_dim(tensor, dim, size):
    return broadcast(tensor.unsqueeze(dim), dim, size)


def add_dim_as_except(tensor, target, *excepts):
    assert len(excepts) == tensor.dim()
    tensor = tensor.clone()
    excepts = [e + target.dim() if e < 0 else e for e in excepts]
    for i in range(target.dim()):
        if i not in excepts:
            tensor.unsqueeze_(i)
    return tensor


def length2mask(lengths, max_length):
    rng = torch.arange(max_length, dtype=lengths.dtype, device=lengths.device)
    lengths = lengths.unsqueeze(-1)
    rng = add_dim_as_except(rng, lengths, -1)
    mask = rng < lengths
    return mask

