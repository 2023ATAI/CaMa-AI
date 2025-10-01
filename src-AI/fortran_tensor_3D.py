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

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Ftensor_3D requires 3 indices (row, col)")
        depth, row, col = key
        if isinstance(depth, torch.Tensor) and (depth.dtype == torch.bool):
            depth_ = torch.where(depth)
            start_depth = self._shift_index(depth_[0], self.start_depth)
            real_row = self._shift_index(row, self.start_row)
            real_col = self._shift_index(col, self.start_col)
        elif isinstance(col, torch.Tensor) and (col.dtype == torch.bool):
            col_ = torch.where(col)
            start_depth = self._shift_index(depth, self.start_depth)
            real_row = self._shift_index(row, self.start_row)
            real_col = self._shift_index(col_[0], self.start_col)
        elif isinstance(row, torch.Tensor) and (row.dtype == torch.bool):
            row_ = torch.where(row)
            start_depth = self._shift_index(depth, self.start_depth)
            real_row = self._shift_index(row_[0], self.start_row)
            real_col = self._shift_index(col, self.start_col)
        else:
            start_depth = self._shift_index(depth, self.start_depth)
            real_row = self._shift_index(row, self.start_row)
            real_col = self._shift_index(col, self.start_col)


        return self.tensor[start_depth, real_row, real_col]

    def __setitem__(self, key, value):
        if not isinstance(key, tuple) or len(key) != 3:
            raise IndexError("Ftensor_3D requires 3 indices (row, col)")
        depth, row, col = key
        start_depth = self._shift_index(depth, self.start_depth)
        real_row = self._shift_index(row, self.start_row)
        real_col = self._shift_index(col, self.start_col)
        self.tensor[start_depth, real_row, real_col] = value

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

    def _normalize_index(self, index):
        # 保证 index 总是 (row, col, depth)
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) == 1:
            return index[0], slice(None), slice(None)
        elif len(index) == 2:
            return index[0], index[1], slice(None)
        elif len(index) == 3:
            return index
        else:
            raise IndexError("Index must have at most 3 elements")

    def _check_bounds(self, row, col, depth):
        # 只有在 int 类型时才检查越界
        if isinstance(row, int):
            if not (0 <= row < self.nrows):
                raise IndexError(f"Row index {row} out of bounds (0, {self.nrows-1})")
        if isinstance(col, int):
            if not (0 <= col < self.ncols):
                raise IndexError(f"Col index {col} out of bounds (0, {self.ncols-1})")
        if isinstance(depth, int):
            if not (0 <= depth < self.ndepths):
                raise IndexError(f"Depth index {depth} out of bounds (0, {self.ndepths-1})")
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
        return (depth + self.start_depth,
                row + self.start_row,
                col + self.start_col)

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
        depth_idx = depth_idx - 1
        row_idx = row_idx - 1
        col_idx = col_idx - 1
        self.tensor.index_put_((depth_idx, row_idx, col_idx), values, accumulate=accumulate)