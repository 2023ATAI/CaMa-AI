import torch
import numpy as np


class Ftensor_1D:
    def __init__(self, tensor: torch.Tensor, start_index=1):
        assert tensor.ndim == 1, "Only 1D tensors are supported"
        self.tensor = tensor
        self.start_index = start_index
        self.length = tensor.shape[0]

    def _shift_index(self, idx):
        if isinstance(idx, int):
            return idx - self.start_index
        if isinstance(idx, slice):
            start = None if idx.start is None else idx.start - self.start_index
            stop = None if idx.stop is None else idx.stop - self.start_index
            return slice(start, stop, idx.step)
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                return torch.where(idx)[0]
            return idx.to(device=self.tensor.device, dtype=torch.long) - self.start_index
        if isinstance(idx, np.ndarray):
            return torch.as_tensor(idx, dtype=torch.long, device=self.tensor.device) - self.start_index
        if isinstance(idx, list):
            return torch.as_tensor(idx, dtype=torch.long, device=self.tensor.device) - self.start_index
        if hasattr(idx, "tensor"):
            return idx.raw().to(device=self.tensor.device, dtype=torch.long) - self.start_index
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitem__(self, index):
        return self.tensor[self._shift_index(index)]

    def __setitem__(self, index, value):
        if isinstance(index, slice):
            start = 0 if index.start is None else index.start - self.start_index
            stop = None if index.stop is None else min(index.stop, self.tensor.size(0))
            self.tensor[slice(start, stop, index.step)] = value
            return
        real_index = self._shift_index(index)
        self.tensor[real_index] = value

    def _check_bounds(self, real_index):
        if not (0 <= real_index < self.length):
            raise IndexError(f"Index out of bounds: converted index {real_index}")

    def shape(self):
        return (self.start_index, self.start_index + self.length - 1)

    def raw(self):
        return self.tensor

    def __repr__(self):
        return f"<Custom1DIndexTensor with index range {self.shape()}>"

    def where(self, condition):
        idx = torch.where(condition)[0]
        return idx + self.start_index

    def __eq__(self, other):
        return self.tensor == other

    def __ne__(self, other):
        return self.tensor != other

    def __lt__(self, other):
        return self.tensor < other

    def __le__(self, other):
        return self.tensor <= other

    def __gt__(self, other):
        return self.tensor > other

    def __ge__(self, other):
        return self.tensor >= other

    def __and__(self, other):
        return self.tensor & other

    def __or__(self, other):
        return self.tensor | other

    def _unwrap(self, other):
        if isinstance(other, Ftensor_1D):
            return other.tensor
        return other
