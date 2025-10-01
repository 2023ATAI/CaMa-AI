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
        elif isinstance(idx, slice):
            start_idx = idx.start - start if idx.start is not None else None
            stop_idx = idx.stop - start if idx.stop is not None else None
            return slice(start_idx, stop_idx, idx.step)
        elif isinstance(idx, torch.Tensor):
            return (idx - start).tolist()
        elif isinstance(idx, np.ndarray):
            return (idx - start).tolist()
        elif isinstance(idx, list):
            return [i - start for i in idx]
        else:
            raise TypeError(f"Unsupported index type: {type(idx)}")

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(row, torch.Tensor) and (row.dtype == torch.bool):
                row_ = torch.where(row)
                real_row = self._shift_index(row_[0], self.start_row)
                real_col = self._shift_index(col, self.start_col)
            elif isinstance(col, torch.Tensor) and (col.dtype == torch.bool):
                col_ = torch.where(col)
                real_row = self._shift_index(row, self.start_row)
                real_col = self._shift_index(col_[0], self.start_col)
            elif hasattr(row, 'tensor') and not hasattr(col, 'tensor'):
                real_row = self._shift_index(row.raw(), self.start_row)
                real_col = self._shift_index(col, self.start_col)
            elif hasattr(col, 'tensor') and not hasattr(row, 'tensor'):
                real_col = self._shift_index(col.raw(), self.start_col)
                real_row = self._shift_index(row, self.start_row)
            else:
                real_row = self._shift_index(row, self.start_row)
                real_col = self._shift_index(col, self.start_col)
        else:
            if isinstance(key, torch.Tensor) and (key.dtype == torch.bool or key.ndim == 2):
                row, col = torch.where(key)
                real_row = row
                real_col =col
            else:
                raise IndexError("Ftensor_2D requires 2 indices (row, col) and bool")
        return self.tensor[real_row, real_col]

    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            row, col = key
            if isinstance(row, torch.Tensor) and (row.dtype == torch.bool):
                row_ = torch.where(row)
                real_row = self._shift_index(row_[0], self.start_row)
                real_col = self._shift_index(col, self.start_col)
            elif isinstance(col, torch.Tensor) and (col.dtype == torch.bool):
                col_ = torch.where(col)
                real_row = self._shift_index(row, self.start_row)
                real_col = self._shift_index(col_[0], self.start_col)
            elif hasattr(row, 'tensor') and not hasattr(col, 'tensor'):
                real_row = self._shift_index(row.raw(), self.start_row)
                real_col = self._shift_index(col, self.start_col)
            elif hasattr(col, 'tensor') and not hasattr(row, 'tensor'):
                real_col = self._shift_index(col.raw(), self.start_col)
                real_row = self._shift_index(row, self.start_row)
            else:
                real_row = self._shift_index(row, self.start_row)
                real_col = self._shift_index(col, self.start_col)

            self.tensor[real_row, real_col]= value
        else:
            if isinstance(key.raw(), torch.Tensor) and key.raw().dtype == torch.bool:
                if key.raw().shape != self.tensor.shape:
                    raise IndexError(f"Boolean mask shape {key.shape} does not match tensor shape {self.tensor.shape}")
                self.tensor[key.raw()] = value
            return



    def _check_bounds(self, real_row, real_col):
        # 如果是 Tensor，判断是否是单个值（标量）
        if torch.is_tensor(real_row) and real_row.numel() == 1:
            real_row = real_row.item()
        if torch.is_tensor(real_col) and real_col.numel() == 1:
            real_col = real_col.item()

        # 如果是数组（非标量），则需要使用逻辑判断的方式
        if torch.is_tensor(real_row) or torch.is_tensor(real_col):
            # 广播判断是否越界
            mask = (
                    (real_row < 0) | (real_row >= self.nrows) |
                    (real_col < 0) | (real_col >= self.ncols)
            )
            if mask.any():
                raise IndexError("Index out of bounds.")
        else:
            # 标量情况直接用 Python 逻辑判断
            if not (0 <= real_row < self.nrows and 0 <= real_col < self.ncols):
                raise IndexError("Index out of bounds.")

    @property
    def shape(self):
        """Return a tuple: (start_row → start_row + nrows - 1, start_col → start_col + ncols - 1)"""
        return (self.start_row, self.start_row + self.nrows - 1,
                self.start_col, self.start_col + self.ncols - 1)

    def raw(self):
        """Return the original tensor"""
        return self.tensor

    def __repr__(self):
        return f"<Custom2DIndexTensor with shape {self.shape}>"

    def where(self, condition):
        row, col = torch.where(condition)
        return (
                row + self.start_row,
                col + self.start_col)

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
        # 将 Fortran-style 索引转为 C-style（从0开始）
        row_idx, col_idx = indices
        row_idx = row_idx - 1
        col_idx = col_idx - 1
        self.tensor.index_put_((row_idx, col_idx), values, accumulate=accumulate)