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
        elif isinstance(idx, slice):
            start = None if idx.start is None else idx.start - self.start_index
            stop = None if idx.stop is None else idx.stop - self.start_index
            return slice(start, stop, idx.step)
        elif isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                return torch.where(idx)[0]  # 取 True 的下标
            else:
                return (idx - self.start_index).tolist()
        elif isinstance(idx, np.ndarray):
            return (idx - self.start_index).tolist()
        elif isinstance(idx, list):
            return [i - self.start_index for i in idx]
        elif hasattr(idx, 'tensor'):
            return (idx.raw() - self.start_index).tolist()
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitem__(self, index):

        real_index = self._shift_index(index)
        return self.tensor[real_index]

    def __setitem__(self, index, value):
        if isinstance(index, int):
            real_index = index - self.start_index
            self.tensor[real_index] = value
        elif isinstance(index, slice):
            start = 0 if index.start is None else index.start - self.start_index
            stop = None if index.stop is None else index.stop
            # 确保 stop 不超过 tensor 的大小
            stop = stop if stop is None else min(stop, self.tensor.size(0))
            # 确保切片大小与 value 匹配
            slice_tensor = self.tensor[slice(start, stop, index.step)]
            if slice_tensor.size(0) == value.size(0):
                self.tensor[slice(start, stop, index.step)] = value
            else:
                raise ValueError(f"Size mismatch: target tensor size is {slice_tensor.size(0)}, but value size is {value.size(0)}")
        elif isinstance(index, torch.Tensor):
            if index.dtype == torch.bool:
                self.tensor[index] = value
            else:
                real_index = self._shift_index(index)
                self.tensor[real_index] = value
        elif hasattr(index, 'tensor'):
            real_index = self._shift_index(index)
            self.tensor[real_index] = value
        else:
            raise TypeError(f"Unsupported index type: {type(index)}")


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
        """Return positions (with adjusted index) where condition is True"""
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