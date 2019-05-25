#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : batched_functional.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 05/16/2019
#
# This file is part of AdvancedIndexing-PyTorch.
# Distributed under terms of the MIT license.

import torch

from .basic_functional import tindex
from .utils import BatchIndicator
from .utils import prod
from .utils import canonize_args, infer_batch_dims, validate_batch_dims
from .utils import is_batched_indexing, is_batched_int_indexing, is_batched_list_indexing, is_batched_bool_indexing, is_int_indexing
from .utils import get_batched_vectorized_new_dim, insert_dim
from .torch_utils import add_dim, add_dim_as_except, length2mask


def batched_index_int(tensor, index, dim):
    batch_dims = index.dim()
    assert dim >= batch_dims

    tensor_shape = tensor.size()
    tensor = tensor.reshape(
        prod(tensor_shape[:batch_dims]), prod(tensor_shape[batch_dims:dim]),
        tensor_shape[dim], prod(tensor_shape[dim+1:])
    )
    index = index.reshape(-1)
    assert tensor.size(0) == index.size(0)

    index = index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    index = index.expand(tensor.size(0), tensor.size(1), 1, tensor.size(3))

    output = tensor.gather(2, index)
    return output.reshape(tensor_shape[:dim] + tensor_shape[dim+1:]), dim, 0


def batched_index_slice(tensor, start, stop, step, dim, padding_zero=True):
    batch_dims = max(tuple(x.dim() for x in [start, stop, step] if is_batched_indexing(x)))

    for x in (start, stop, step):
        if is_batched_indexing(x):
            assert batch_dims == x.dim()
            assert tensor.size()[:batch_dims] == x.size()

    assert dim >= batch_dims

    tensor_shape = tensor.size()
    tensor = tensor.reshape(
        prod(tensor_shape[:batch_dims]), prod(tensor_shape[batch_dims:dim]),
        tensor_shape[dim], prod(tensor_shape[dim+1:])
    )

    def canonize_index(x, default):
        if x is None:
            return default
        elif isinstance(x, int):
            return x
        elif torch.is_tensor(x):
            return x.reshape(-1)

    start = canonize_index(start, 0)
    stop = canonize_index(stop, tensor_shape[dim])
    step = canonize_index(step, 1)

    step_is_positive = -1 + 2 * (step > 0)
    if torch.is_tensor(step_is_positive):
        step_is_positive = step_is_positive.long()
    seg_lengths = (stop - start - step_is_positive) // step + 1
    max_seg_length = seg_lengths.max().item()
    mask = length2mask(seg_lengths, max_seg_length)

    # indices values: [[0, 1, 0, 0, ...], [1, 2, 3, 0, ...], ...]
    # NB(Jiayuan Mao @ 05/24): torch does not support cuda-side arange.
    base = torch.arange(max_seg_length, dtype=torch.long).to(device=tensor.device)
    if torch.is_tensor(step):
        step = step.unsqueeze(-1)
    if torch.is_tensor(start):
        start = start.unsqueeze(-1)
    indices = base.unsqueeze(0) * step + start
    indices = indices * mask.long()  # shape: [batch_size, max_seg_length]

    indices = indices.unsqueeze(1).unsqueeze(-1)
    indices = indices.expand(tensor.size(0), tensor.size(1), max_seg_length, tensor.size(3))
    output = tensor.gather(2, indices)

    if padding_zero:
        output = output * add_dim_as_except(mask.type_as(output), output, 0, 2)

    return output.reshape(tensor_shape[:dim] + (max_seg_length, ) + tensor_shape[dim+1:]), dim, seg_lengths.reshape(tensor_shape[:batch_dims])


def batched_index_vector_dim(tensor, indices, indices_length, dim, padding_zero=True):
    batch_dims = indices.dim() - 1
    assert tensor.size()[:batch_dims] == indices.size()[:batch_dims]
    assert dim >= batch_dims

    max_indices_length = indices.size(-1)
    if indices_length is not None:
        mask = length2mask(indices_length.reshape(-1), max_indices_length)

    tensor_shape = tensor.size()
    tensor = tensor.reshape(
        prod(tensor_shape[:batch_dims]), prod(tensor_shape[batch_dims:dim]),
        tensor_shape[dim], prod(tensor_shape[dim+1:])
    )
    indices_shape = indices.size()
    indices = indices.reshape(-1, max_indices_length)
    indices = indices.unsqueeze(1).unsqueeze(-1)
    indices = indices.expand(tensor.size(0), tensor.size(1), max_indices_length, tensor.size(3))
    output = tensor.gather(2, indices)

    if padding_zero and indices_length is not None:
        output = output * add_dim_as_except(mask.type_as(output), output, 0, 2)

    if indices_length is None:
        indices_length = torch.zeros(indices_shape[:-1], dtype=torch.long, device=tensor.device) + max_indices_length

    return output.reshape(tensor_shape[:dim] + (max_indices_length, ) + tensor_shape[dim+1:]), dim, indices_length


def batched_index_vectors(tensor, indices, indices_length, dims, padding_zero=True):
    batch_dims = indices[0].dim() - 1
    dims = tuple(dims)

    assert len(dims) == len(indices)
    assert all(i.size() == indices[0].size() for i in indices)
    assert indices[0].size()[:-1] == tensor.size()[:batch_dims]

    max_indices_length = indices[0].size(-1)
    if indices_length is not None:
        mask = length2mask(indices_length.reshape(-1), max_indices_length)

    permute_dims = tuple(range(batch_dims)) + dims + tuple(i for i in range(batch_dims, tensor.dim()) if i not in dims)
    tensor = tensor.permute(permute_dims)
    tensor_shape = tensor.size()
    tensor = tensor.reshape(
        prod(tensor_shape[:batch_dims]), prod(tensor_shape[batch_dims:batch_dims + len(dims)]),
        prod(tensor_shape[batch_dims + len(dims):])
    )

    max_sizes = tensor_shape[batch_dims:batch_dims + len(dims)]
    index = 0
    stride = 1
    for i, s in reversed(list(zip(indices, max_sizes))):
        index += i.reshape(-1, max_indices_length) * stride
        stride *= s

    index = index.unsqueeze(-1)
    index = index.expand(index.size(0), index.size(1), tensor.size(2))
    output = tensor.gather(1, index)

    if padding_zero and indices_length is not None:
        output = output * add_dim_as_except(mask.type_as(output), output, 0, 1)

    if indices_length is None:
        indices_length = torch.zeros(indices[0].size()[:-1], dtype=torch.long, device=tensor.device) + max_indices_length

    return output.reshape(tensor_shape[:batch_dims] + (max_indices_length, ) + tensor_shape[batch_dims + len(dims):]), batch_dims, indices_length


def _basic_batched_index(value, args, padding_zero=True):
    args = canonize_args(args, value.dim())
    batch_dims = infer_batch_dims(args)

    assert value.dim() > batch_dims
    validate_batch_dims(args, batch_dims)

    new_args = list()
    advanced_indices = list()
    advanced_indices_dim = list()
    j = 0
    for i, arg in enumerate(args):
        if isinstance(arg, BatchIndicator):
            new_args.append(slice(None, None, None))
            j += 1
        elif is_batched_indexing(arg):
            new_args.append(slice(None, None, None))
            advanced_indices.append(arg)
            advanced_indices_dim.append(j)
            j += 1
        else:
            new_args.append(arg)
            if is_int_indexing(arg):
                pass
            else:
                j += 1

    output = tindex(value, tuple(new_args))
    rest_dims = output.dim() - batch_dims
    rest_shape = output.size()[batch_dims:]
    if rest_dims == 0:
        shape = torch.zeros(output.size()[:batch_dims], dtype=torch.long, device=output.device)
        assert len(advanced_indices) == 0
    else:
        shape = torch.tensor(rest_shape)
        for i, size in enumerate(output.size()[:batch_dims]):
            shape = add_dim(shape, i, size)

    nr_advance_indices = len(advanced_indices)
    marked = [False for i in range(nr_advance_indices)]
    for i in range(nr_advance_indices):
        arg, dim = advanced_indices[i], advanced_indices_dim[i]
        if is_batched_int_indexing(arg, batch_dims):
            output, _, _ = batched_index_int(output, arg, dim)
            to_delete = dim - batch_dims

            if shape.size(-1) == 1:
                shape = torch.zeros(output.size()[:batch_dims], dtype=torch.long, device=output.device)
            elif to_delete == 0:
                shape = shape[..., 1:]
            elif to_delete == shape.size(-1) - 1:
                shape = shape[..., :-1]
            else:
                shape = torch.cat((shape[..., :to_delete], shape[..., to_delete + 1:]), dim=-1)

            for j in range(i + 1, nr_advance_indices):
                advanced_indices_dim[j] -= 1
            marked[i] = True
        elif isinstance(arg, slice):
            output, _, new_size = batched_index_slice(output, arg.start, arg.stop, arg.step, dim, padding_zero=padding_zero)
            # NB(Jiayuan Mao @ 05/25): a bug with PyTorch: without the contiguous the set_index will behave wrong.
            shape = shape.contiguous()
            shape[..., dim - batch_dims] = new_size
            marked[i] = True

    rest_advanced_indices = [arg for i, arg in enumerate(advanced_indices) if not marked[i]]
    rest_advanced_indices_dim = [arg for i, arg in enumerate(advanced_indices_dim) if not marked[i]]

    return output, shape, batch_dims, rest_advanced_indices, rest_advanced_indices_dim


def boindex(value, args, indices_length=None, padding_zero=True):
    output, shape, batch_dims, ai, aid = _basic_batched_index(value, args, padding_zero=padding_zero)

    for i, dim in zip(ai, aid):
        if is_batched_bool_indexing(i, batch_dims):
            raise NotImplementedError()

        output, _, new_size = batched_index_vector_dim(output, i, indices_length, dim, padding_zero=padding_zero)
        shape = shape.contiguous()
        shape[..., dim - batch_dims] = new_size

    return output, shape


def bvindex(value, args, indices_length=None, padding_zero=True):
    output, shape, batch_dims, ai, aid = _basic_batched_index(value, args, padding_zero=padding_zero)

    if len(ai) == 0:
        return output, shape

    for i, dim in zip(ai, aid):
        if is_batched_bool_indexing(i, batch_dims):
            raise NotImplementedError()

    rest_dims = tuple(i for i in range(shape.size(-1)) if i + batch_dims not in aid)
    output, _, new_size = batched_index_vectors(output, ai, indices_length, aid, padding_zero=padding_zero)

    if len(rest_dims) == 0:
        shape = new_size.unsqueeze(-1)
    else:
        shape = torch.cat((new_size.unsqueeze(-1), shape[..., rest_dims]), dim=-1)

    return output, shape


def btindex(value, args, indices_length=None, padding_zero=True):
    args = canonize_args(args, value.dim())
    batch_dims = infer_batch_dims(args)

    output, shape = bvindex(value, args, indices_length, padding_zero=padding_zero)

    new_dim = get_batched_vectorized_new_dim(args, batch_dims)
    if new_dim != -1:
        if new_dim != batch_dims:
            output = insert_dim(output, batch_dims, new_dim)
            shape = shape.contiguous()
            shape[..., 0], shape[..., new_dim - batch_dims] = shape[..., new_dim - batch_dims], shape[..., 0]

    return output, shape


def bfindex(value, args, indices_length=None, padding_zero=True):
    args = canonize_args(args, value.dim())
    batch_dims = infer_batch_dims(args)

    output, shape = bvindex(value, args, indices_length, padding_zero=padding_zero)

    new_dim = get_batched_vectorized_new_dim(args, batch_dims, True)
    if new_dim != -1:
        if new_dim != batch_dims:
            output = insert_dim(output, batch_dims, new_dim)
            shape = shape.contiguous()
            shape[..., 0], shape[..., new_dim - batch_dims] = shape[..., new_dim - batch_dims], shape[..., 0]

    return output, shape

