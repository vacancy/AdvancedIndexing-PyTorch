#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/15/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

from .utils import batch
from . import basic_functional
from . import batched_functional

__all__ = ['batch', 'tindex', 'findex', 'vindex', 'oindex', 'btindex', 'bfindex', 'bvindex', 'boindex']


class IndexWrapper(object):
    __index_func__ = None

    def __init__(self, tensor):
        self.tensor = tensor

    def __getitem__(self, args):
        return type(self).__index_func__(self.tensor, args)


class BatchedIndexWrapper(object):
    __index_func__ = None

    def __init__(self, tensor, indices_length=None, padding_zero=True):
        self.tensor = tensor
        self.indices_length = indices_length
        self.padding_zero = padding_zero

    def __call__(self, indices_length=None, padding_zero=None):
        if indices_length is not None:
            self.indices_length = indices_length
        if padding_zero is not None:
            self.padding_zero = padding_zero
        return self

    def __getitem__(self, args):
        return type(self).__index_func__(self.tensor, args, indices_length=self.indices_length, padding_zero=self.padding_zero)


class tindex(IndexWrapper):
    __index_func__ = basic_functional.tindex


class findex(IndexWrapper):
    __index_func__ = basic_functional.findex


class vindex(IndexWrapper):
    __index_func__ = basic_functional.vindex


class oindex(IndexWrapper):
    __index_func__ = basic_functional.oindex


class btindex(BatchedIndexWrapper):
    __index_func__ = batched_functional.btindex


class bfindex(BatchedIndexWrapper):
    __index_func__ = batched_functional.bfindex


class bvindex(BatchedIndexWrapper):
    __index_func__ = batched_functional.bvindex


class boindex(BatchedIndexWrapper):
    __index_func__ = batched_functional.boindex

