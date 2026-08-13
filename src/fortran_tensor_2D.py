import torch
import numpy as np


class Ftensor_2D:
    def __init__(self, tensor: torch.Tensor, start_row=1, start_col=1):
        assert tensor.ndim == 2, "Only 2D tensors are supported"
        self.tensor = tensor
        self.start_row = start_row
        self.start_col = start_col
        self.nrows, self.ncols = tensor.shape

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
        if isinstance(key, tuple):
            row, col = key
            real_row = self._shift_index(row, self.start_row)
            real_col = self._shift_index(col, self.start_col)
            return self.tensor[real_row, real_col]
        if isinstance(key, torch.Tensor) and (key.dtype == torch.bool or key.ndim == 2):
            row, col = torch.where(key)
            return self.tensor[row, col]
        if hasattr(key, "raw") and key.raw().dtype == torch.bool:
            row, col = torch.where(key.raw())
            return self.tensor[row, col]
        raise IndexError("Ftensor_2D requires 2 indices (row, col) or a bool mask")

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            real_row = self._shift_index(row, self.start_row)
            real_col = self._shift_index(col, self.start_col)
            self.tensor[real_row, real_col] = value
            return
        if hasattr(key, "raw") and key.raw().dtype == torch.bool:
            if key.raw().shape != self.tensor.shape:
                raise IndexError(f"Boolean mask shape {key.shape} does not match tensor shape {self.tensor.shape}")
            self.tensor[key.raw()] = value
            return
        if isinstance(key, torch.Tensor) and key.dtype == torch.bool:
            self.tensor[key] = value
            return
        raise IndexError("Ftensor_2D requires 2 indices (row, col) or a bool mask")

    def _check_bounds(self, real_row, real_col):
        if torch.is_tensor(real_row) or torch.is_tensor(real_col):
            mask = (
                (real_row < 0) | (real_row >= self.nrows) |
                (real_col < 0) | (real_col >= self.ncols)
            )
            if torch.any(mask):
                raise IndexError("Index out of bounds.")
        elif not (0 <= real_row < self.nrows and 0 <= real_col < self.ncols):
            raise IndexError("Index out of bounds.")

    @property
    def shape(self):
        return (self.start_row, self.start_row + self.nrows - 1,
                self.start_col, self.start_col + self.ncols - 1)

    def raw(self):
        return self.tensor

    def __repr__(self):
        return f"<Custom2DIndexTensor with shape {self.shape}>"

    def where(self, condition):
        row, col = torch.where(condition)
        return row + self.start_row, col + self.start_col

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
        if isinstance(other, Ftensor_2D):
            return other.tensor
        return other

    def index_put_(self, indices, values, accumulate=False):
        row_idx, col_idx = indices
        row_idx = row_idx.to(device=self.tensor.device, dtype=torch.long) - 1
        col_idx = col_idx.to(device=self.tensor.device, dtype=torch.long) - 1
        self.tensor.index_put_((row_idx, col_idx), values, accumulate=accumulate)
