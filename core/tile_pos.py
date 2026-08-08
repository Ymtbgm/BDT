import math
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional


# 基于 levels.json 统计的精确尺寸到 view 映射（合并相近 view，差值 < 0.05）
_VIEW_MAP_BY_SIZE = {
    (4, 9): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (5, 9): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (5, 10): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (5, 11): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (5, 13): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (6, 7): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (6, 8): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (6, 9): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (6, 10): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (6, 11): ((0.0, -5.08, -8.02), (0.643, -5.58, -8.898)),
    (6, 12): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (6, 13): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (7, 8): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (7, 9): ((0.0, -4.81, -7.74), (0.594, -5.31, -8.622)),
    (7, 10): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (7, 11): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (7, 12): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (7, 13): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (7, 14): ((-0.5, -6.6, -10.63), (0.603, -7.1, -11.555)),
    (8, 9): ((0.0, -5.08, -8.02), (0.643, -5.58, -8.898)),
    (8, 10): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (8, 11): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (8, 12): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (8, 13): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (8, 14): ((0.0, -5.6, -8.9), (0.795, -6.1, -9.765)),
    (8, 15): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (8, 17): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (8, 30): ((-9.0, -7.1, -11.5), (-7.616, -7.6, -13.888)),
    (8, 32): ((-9.5, -6.1, -10.63), (-8.26, -6.6, -13.118)),
    (9, 8): ((0.0, -6.85, -11.065), (1.171, -7.35, -11.897)),
    (9, 9): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 10): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 11): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 12): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 13): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (9, 14): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 15): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (9, 17): ((0.0, -8.1, -11.5), (1.247, -8.6, -12.325)),
    (9, 20): ((-3.0, -5.6, -8.9), (-2.159, -6.1, -10.286)),
    (9, 21): ((-1.5, -6.85, -11.065), (-0.306, -7.35, -12.157)),
    (9, 24): ((-6.0, -6.85, -11.065), (-4.737, -7.35, -12.939)),
    (9, 30): ((-8.5, -6.6, -10.63), (-7.275, -7.1, -12.945)),
    (10, 12): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (10, 13): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (10, 14): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (10, 15): ((0.0, -7.1, -10.63), (1.096, -7.6, -11.469)),
    (10, 17): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (11, 12): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (11, 13): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (11, 14): ((0.0, -6.1, -9.78), (0.948, -6.6, -10.631)),
    (11, 15): ((0.0, -7.1, -11.5), (1.247, -7.6, -12.325)),
    (11, 16): ((-0.5, -7.6, -10.63), (0.603, -8.1, -11.555)),
    (11, 17): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
    (13, 17): ((-0.5, -7.6, -10.63), (0.603, -8.1, -11.555)),
    (17, 11): ((0.0, -11.6, -11.5), (1.247, -12.1, -12.325)),
    (19, 21): ((-4.5, -10.31, -7.74), (-3.838, -10.81, -9.404)),
    (31, 37): ((0.0, -7.1, -11.5), (1.247, -7.6, -12.325)),
    (32, 33): ((0.0, -7.1, -11.5), (1.247, -7.6, -12.325)),
    (40, 40): ((0.0, -6.6, -10.63), (1.096, -7.1, -11.469)),
}
_DEFAULT_VIEW_NORMAL = (0.0, -6.1, -9.78)
_DEFAULT_VIEW_SIDE = (0.948, -6.6, -10.631)


def _guess_view(grid_cols: int, grid_rows: int) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """根据地图尺寸从精确映射表查询 view，未命中则返回默认值。"""
    key = (grid_rows, grid_cols)
    if key in _VIEW_MAP_BY_SIZE:
        return _VIEW_MAP_BY_SIZE[key]
    return _DEFAULT_VIEW_NORMAL, _DEFAULT_VIEW_SIDE


from core.paths import game_data


def _levels_json_path() -> Path:
    """返回 levels.json 的查找路径。"""
    return game_data("levels.json")


def _load_levels() -> list:
    """加载并缓存 levels.json 内容。"""
    p = _levels_json_path()
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _load_view_from_json(code: Optional[str] = None, name: Optional[str] = None) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """从本地 levels.json 加载精确 view。"""
    for lv in _load_levels():
        if code and lv.get("code") == code:
            view = lv.get("view", [])
            if len(view) >= 2:
                return tuple(view[0]), tuple(view[1])
        if name and lv.get("name") == name:
            view = lv.get("view", [])
            if len(view) >= 2:
                return tuple(view[0]), tuple(view[1])
    return None


def load_stage_dimensions(code: str) -> Optional[Tuple[int, int]]:
    """根据关卡 code 从 levels.json 加载地图尺寸 (width, height)。

    返回 (grid_cols, grid_rows)，即列数、行数；未找到时返回 None。
    """
    for lv in _load_levels():
        if lv.get("code") == code:
            width = lv.get("width")
            height = lv.get("height")
            if isinstance(width, int) and isinstance(height, int):
                return width, height
    return None


class TilePosCalculator:
    """基于 3D 矩阵变换精确计算地图格子屏幕坐标。"""

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        grid_rows: int,
        grid_cols: int,
        view_normal: Optional[Tuple[float, float, float]] = None,
        view_side: Optional[Tuple[float, float, float]] = None,
        stage_code: Optional[str] = None,
        stage_name: Optional[str] = None,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.ratio = screen_height / screen_width

        # 优先使用用户传入的 view，其次从 levels.json 查询，最后按尺寸猜测
        if view_normal is not None:
            self.view_normal = view_normal
            self.view_side = view_side or _guess_view(grid_cols, grid_rows)[1]
        else:
            loaded = _load_view_from_json(stage_code, stage_name)
            if loaded:
                self.view_normal, self.view_side = loaded
            else:
                self.view_normal, self.view_side = _guess_view(grid_cols, grid_rows)

        self._init_matrices()

    def _init_matrices(self):
        self.matrix_p = np.array([
            [self.ratio / math.tan(math.pi * 20 / 180), 0, 0, 0],
            [0, 1 / math.tan(math.pi * 20 / 180), 0, 0],
            [0, 0, -(1000 + 0.3) / (1000 - 0.3), -(1000 * 0.3 * 2) / (1000 - 0.3)],
            [0, 0, -1, 0],
        ], dtype=np.float64)

        self.matrix_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(math.pi * 30 / 180), -math.sin(math.pi * 30 / 180), 0],
            [0, -math.sin(math.pi * 30 / 180), -math.cos(math.pi * 30 / 180), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        self.matrix_y = np.array([
            [math.cos(math.pi * 10 / 180), 0, math.sin(math.pi * 10 / 180), 0],
            [0, 1, 0, 0],
            [-math.sin(math.pi * 10 / 180), 0, math.cos(math.pi * 10 / 180), 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

    def _adapter(self) -> Tuple[float, float]:
        from_ratio = 9 / 16
        to_ratio = 3 / 4
        if self.ratio < from_ratio - 0.00001:
            return 0.0, 0.0
        t = (self.ratio - from_ratio) / (to_ratio - from_ratio)
        return -1.4 * t, -2.8 * t

    def _get_transform_matrix(self, side: bool = False):
        adapter_y, adapter_z = self._adapter()
        if side:
            vx, vy, vz = self.view_side
        else:
            vx, vy, vz = self.view_normal
        vy += adapter_y
        vz += adapter_z

        raw = np.array([
            [1, 0, 0, -vx],
            [0, 1, 0, -vy],
            [0, 0, 1, -vz],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        if side:
            matrix = np.dot(self.matrix_x, self.matrix_y)
            matrix = np.dot(matrix, raw)
        else:
            matrix = np.dot(self.matrix_x, raw)
        return np.dot(self.matrix_p, matrix)

    def get_screen_pos(self, row: int, col: int, side: bool = False) -> Tuple[int, int]:
        h, w = self.grid_rows, self.grid_cols
        wx = col - (w - 1) / 2
        wy = (h - 1) / 2 - row
        wz = 0.0

        matrix = self._get_transform_matrix(side)
        px, py, _, pw = np.dot(matrix, np.array([wx, wy, wz, 1]))

        sx = (1 + px / pw) / 2 * self.screen_width
        sy = (1 - py / pw) / 2 * self.screen_height
        return int(sx), int(sy)

    def get_all_positions(self, side: bool = False):
        result = []
        for r in range(self.grid_rows):
            row = []
            for c in range(self.grid_cols):
                row.append(self.get_screen_pos(r, c, side))
            result.append(row)
        return result

    def get_tile_polygon(
        self, row: int, col: int, side: bool = False
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """返回指定格子在屏幕上的投影四边形（左上/右上/右下/左下顺序不严格）。

        基于格子在世界坐标中占据 [col-0.5, col+0.5] x [row-0.5, row+0.5]
        的矩形，投影四个角到屏幕。
        """
        matrix = self._get_transform_matrix(side)

        corners = [
            (col - 0.5, row - 0.5),
            (col + 0.5, row - 0.5),
            (col + 0.5, row + 0.5),
            (col - 0.5, row + 0.5),
        ]
        screen_corners = []
        for wx, wy in corners:
            sx_world = wx - (self.grid_cols - 1) / 2
            sy_world = (self.grid_rows - 1) / 2 - wy
            px, py, _, pw = np.dot(matrix, np.array([sx_world, sy_world, 0.0, 1]))
            sx = (1 + px / pw) / 2 * self.screen_width
            sy = (1 - py / pw) / 2 * self.screen_height
            screen_corners.append((int(sx), int(sy)))
        return tuple(screen_corners)  # type: ignore[return-value]

    def hit_test(
        self, x: int, y: int, side: bool = False
    ) -> Optional[Tuple[int, int]]:
        """判断屏幕点 (x, y) 落在哪个格子的投影四边形内部。

        优先返回包含该点的格子；若点恰好落在多个四边形重叠处（side 视角边缘），
        取中心点最近的一个；若完全不命中，返回 None。
        """
        import cv2

        point = (float(x), float(y))
        candidates = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                poly = np.array(self.get_tile_polygon(r, c, side), dtype=np.float32)
                dist = cv2.pointPolygonTest(poly, point, False)
                if dist >= 0:
                    cx, cy = self.get_screen_pos(r, c, side)
                    d = (cx - x) ** 2 + (cy - y) ** 2
                    candidates.append((d, r, c))
        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates[0][1], candidates[0][2]
        return None
