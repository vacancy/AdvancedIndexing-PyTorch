#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/15/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.


from . import basic_functional
from .utils import IndexWrapper

__all__ = ['tindex', 'findex', 'vindex', 'oindex']


class tindex(IndexWrapper):
    __index_func__ = basic_functional.tindex


class findex(IndexWrapper):
    __index_func__ = basic_functional.findex


class vindex(IndexWrapper):
    __index_func__ = basic_functional.vindex


class oindex(IndexWrapper):
    __index_func__ = basic_functional.oindex

