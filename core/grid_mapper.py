from typing import Tuple, Optional
import numpy as np
from core.tile_pos import TilePosCalculator


class GridMapper:
    def __init__(self, window_width: int, window_height: int, grid_rows: int, grid_cols: int,
                 use_precise: bool = True, stage_code: Optional[str] = None, stage_name: Optional[str] = None):
        self.window_width = window_width
        self.window_height = window_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self._use_precise = use_precise
        self._stage_code = stage_code
        self._stage_name = stage_name
        self._precise_calc: TilePosCalculator | None = None
        self._positions_normal: list[list[Tuple[int, int]]] = []
        self._positions_side: list[list[Tuple[int, int]]] = []
        self._cell_w = 0.0
        self._cell_h = 0.0
        self._recalculate()

    def set_window_size(self, width: int, height: int):
        self.window_width = width
        self.window_height = height
        self._recalculate()

    def _recalculate(self):
        if self._use_precise:
            self._precise_calc = TilePosCalculator(
                self.window_width, self.window_height,
                self.grid_rows, self.grid_cols,
                stage_code=self._stage_code,
                stage_name=self._stage_name,
            )
            self._positions_normal = self._precise_calc.get_all_positions(side=False)
            self._positions_side = self._precise_calc.get_all_positions(side=True)
        else:
            self._precise_calc = None
            self._positions_normal = []
            self._positions_side = []

        self._cell_w = self.window_width / self.grid_cols
        self._cell_h = self.window_height / self.grid_rows

    def grid_to_pixel(self, row: int, col: int, side: bool = False) -> Tuple[int, int]:
        positions = self._positions_side if side else self._positions_normal
        if self._use_precise and positions:
            r = max(0, min(row, self.grid_rows - 1))
            c = max(0, min(col, self.grid_cols - 1))
            return positions[r][c]
        x = int((col + 0.5) * self._cell_w)
        y = int((row + 0.5) * self._cell_h)
        return x, y

    def pixel_to_grid(self, x: int, y: int) -> Tuple[int, int]:
        if self._use_precise and self._positions:
            # 精确模式下格子大小不均匀，遍历找最近点
            best_r, best_c = 0, 0
            best_dist = float("inf")
            for r in range(self.grid_rows):
                for c in range(self.grid_cols):
                    px, py = self._positions[r][c]
                    dist = (px - x) ** 2 + (py - y) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_r, best_c = r, c
            return best_r, best_c
        col = int(x / self._cell_w)
        row = int(y / self._cell_h)
        col = max(0, min(col, self.grid_cols - 1))
        row = max(0, min(row, self.grid_rows - 1))
        return row, col

    def get_grid_size(self) -> Tuple[float, float]:
        return self._cell_w, self._cell_h
