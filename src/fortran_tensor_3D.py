import torch
import numpy as np


class Ftensor_3D:
    def __init__(self, tensor: torch.Tensor, start_depth=1, start_row=1, start_col=1):
        assert tensor.ndim == 3, "Only 3D tensors are supported"
        self.tensor = tensor
        self.start_depth = start_depth
        self.start_row = start_row
        self.start_col = start_col
        self.ndepths, self.nrows, self.ncols = tensor.shape

    def _shift_index(self, idx, start):
        if isinstance(idx, int):
            return idx - start
        if isinstance(idx, slice):
            start_idx = idx.start - start if idx.start is not None else None
            stop_idx = idx.stop - start if idx.stop is not None else None
            return slice(start_idx, stop_idx, idx.step)
        if isinstance(idx, torch.Tensor):
            if idx.dtype == torch.bool:
                return torch.where(idx)[0]
            return idx.to(device=self.tensor.device, dtype=torch.long) - start
        if isinstance(idx, np.ndarray):
            return torch.as_tensor(idx, dtype=torch.long, device=self.tensor.device) - start
        if isinstance(idx, list):
            return torch.as_tensor(idx, dtype=torch.long, device=self.tensor.device) - start
        if hasattr(idx, "tensor"):
            return idx.raw().to(device=self.tensor.device, dtype=torch.long) - start
        raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Ftensor_3D requires 3 indices (depth, row, col)")
        depth, row, col = key
        real_depth = self._shift_index(depth, self.start_depth)
        real_row = self._shift_index(row, self.start_row)
        real_col = self._shift_index(col, self.start_col)
        return self.tensor[real_depth, real_row, real_col]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Ftensor_3D requires 3 indices (depth, row, col)")
        depth, row, col = key
        real_depth = self._shift_index(depth, self.start_depth)
        real_row = self._shift_index(row, self.start_row)
        real_col = self._shift_index(col, self.start_col)
        self.tensor[real_depth, real_row, real_col] = value

    def _normalize_index(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) == 1:
            return index[0], slice(None), slice(None)
        if len(index) == 2:
            return index[0], index[1], slice(None)
        if len(index) == 3:
            return index
        raise IndexError("Index must have at most 3 elements")

    def _check_bounds(self, row, col, depth):
        if isinstance(row, int) and not (0 <= row < self.nrows):
            raise IndexError(f"Row index {row} out of bounds (0, {self.nrows - 1})")
        if isinstance(col, int) and not (0 <= col < self.ncols):
            raise IndexError(f"Col index {col} out of bounds (0, {self.ncols - 1})")
        if isinstance(depth, int) and not (0 <= depth < self.ndepths):
            raise IndexError(f"Depth index {depth} out of bounds (0, {self.ndepths - 1})")

    @property
    def shape(self):
        return (self.start_depth, self.start_depth + self.ndepths - 1,
                self.start_row, self.start_row + self.nrows - 1,
                self.start_col, self.start_col + self.ncols - 1)

    def raw(self):
        return self.tensor

    def __repr__(self):
        return f"<Custom3DIndexTensor with shape {self.shape}>"

    def where(self, condition):
        depth, row, col = torch.where(condition)
        return depth + self.start_depth, row + self.start_row, col + self.start_col

    def __eq__(self, other): return self.tensor == other
    def __ne__(self, other): return self.tensor != other
    def __lt__(self, other): return self.tensor < other
    def __le__(self, other): return self.tensor <= other
    def __gt__(self, other): return self.tensor > other
    def __ge__(self, other): return self.tensor >= other
    def __and__(self, other): return self.tensor & other
    def __or__(self, other): return self.tensor | other

    def _unwrap(self, other):
        if isinstance(other, Ftensor_3D):
            return other.tensor
        return other

    def index_put_(self, indices, values, accumulate=False):
        depth_idx, row_idx, col_idx = indices
        depth_idx = depth_idx.to(device=self.tensor.device, dtype=torch.long) - 1
        row_idx = row_idx.to(device=self.tensor.device, dtype=torch.long) - 1
        col_idx = col_idx.to(device=self.tensor.device, dtype=torch.long) - 1
        self.tensor.index_put_((depth_idx, row_idx, col_idx), values, accumulate=accumulate)
