# AdvancedIndexing-PyTorch
(Batched) advanced indexing for PyTorch.

The `torch_index` package is designed for performing advanced indexing on PyTorch tensors. Beyond the support of basic indexing methods (vectorized indexing, outer indexing, numpy-style indexing, pytorch-style indexing), it also supports **batched** indexing. That is, the indices to the tensor may vary across different batch index.

Example is better than precept!

```python
import torch
from torch_index import batch
from torch_index import btindex

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

btindex(a)[batch, batch, blist, 5:bint0:-1, blist]
```

Another way to use the package is to run the `torch_index.patch_torch()` function.
```python
import torch_index; torch_index.patch_torch()
a.btindex[batch, batch, blist, 5:bint0:-1, blist]
```

For more examples and tests, see the `tests/` directory. Run `nosetests` to run all tests.

## Requirements
- pytorch > 1.0.1

To run the tests:
- numpy
- nose

## Supported Indexing Methods

- `torch_index.tindex` (torch_index): PyTorch-compatible advanced indexing, with extra negative step support.
- `torch_index.findex` (fancy_index): numpy-compatible fancy indexing. This indexing method differs slightly with the PyTorch-style indexing when jointly handling vectorized indexing and single-int indexing.
```python
>>> import torch
>>> import numpy as np
>>> np.zeros((5, 6, 7, 8))[:, [0], :, 0].shape
(1, 5, 7)
>>> torch.zeros((5, 6, 7, 8))[:, [0], :, 0].shape
torch.Size([5, 1, 7])
```
- `torch_index.vindex` (vectorized_index): vectorized indexing. If there are vectorized indices, the new dimension will always be added to the front.
- `torch_index.oindex` (outer_index): outer indexing. If there are multiple vectorized indices, the result is the product of all vectorized indices.
```python
>>> import torch
>>> import torch_index
>>> torch_index.vindex(torch.zeros(5, 6, 7, 8))[:, [0, 1], [0, 1]].shape
torch.Size([2, 5, 8])
>>> torch_index.oindex(torch.zeros(5, 6, 7, 8))[:, [0, 1], [0, 1]].shape
torch.Size([5, 2, 2, 8])
```
For more details, please refer to this [numpy proposal page](https://www.numpy.org/neps/nep-0021-advanced-indexing.html).
- `torch_index.btindex` (batched_torch_index).
- `torch_index.bfindex` (batched_fancy_index).
- `torch_index.bvindex` (batched_vectorized_index).
- `torch_index.boindex` (batched_outer_index).

The batched indexing methods always starts from specifying multiple **leading** dimensions as "batch dimensions". If the batch dimensions are not the first K dimensions, please permute the dimensions first.
```python
>>> import torch
>>> import torch_index
>>> a = torch.rand(2, 3, 10)
```
Suppose that the first two dimensions are batch dimensions, then you can use:
```python
>>> output, shape = torch_index.btindex(a)[torch_index.batch, torch_index.batch, torch.zeros((2, 3), dtype=torch.long)]
>>> output.shape
torch.Size([2, 3])
>>> shape
tensor([[0, 0, 0],
        [0, 0, 0]])
```
This is equivalent to performing indexing (`tindex`) for all data items in the batch, and concatenate the results along the batch dimensions.

There are two return values for all batched indexing methods. The first one is the concatenated results. The second one is a long tensor of shape `[2, 3, rest_dimensions]`, indicating the size of indexing output for each of the item in the batch.
```python
>>> start = torch.tensor([[0, 0, 0], [1, 2, 3]])
>>> stop = torch.tensor([[3, 4, 5], [4, 5, 6]])
>>> output, shape = torch_index.btindex(a)[torch_index.batch, torch_index.batch, start:stop]
>>> shape
tensor([[[3], [4], [5]],
        [[3], [3], [3]]])
```

The following types of batched indices are supported:
- batched int indexing: the index is of shape `[2, 3]`
- batched slice indexing: the index is `start:stop:step`, where `start`, `stop`, and `step` can be tensors of shape `[2, 3]`, or int, or None.
- batched vector indexing: the index is of shape `[2, 3, K]`

The following types of batched indices are **NOT** supported:
- batched bool indexing.

When you are specifying vector indices, you can also specify a list indicating the length of vectors across different items in a batch. The length vector is of shape `[2, 3]`
```python
>>> output, shape = torch_index.btindex(a)[torch_index.batch, torch_index.batch, torch.zeros((2, 3, 5), dtype=torch.long)]
>>> shape
tensor([[[5], [5], [5]],
        [[5], [5], [5]]])
>>> vec_length = torch.tensor([[3, 4, 5], [3, 4, 5]])
>>> output, shape = torch_index.btindex(a, vec_length)[torch_index.batch, torch_index.batch, torch.zeros((2, 3, 5), dtype=torch.long)]
>>> shape
tensor([[[3], [4], [5]],
        [[3], [4], [5]]])
```

## Authors and License

Copyright (c) 2019-, [Jiayuan Mao](https://jiayuanm.com).

Distributed under **MIT License** (See LICENSE)
