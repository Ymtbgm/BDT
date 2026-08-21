import math
import numpy as np
import json
from pathlib import Path
from typing import Tuple, Optional


# 当 buildableType 缺失时，用 tileKey 判断格子是否可部署
_NON_DEPLOYABLE_TILE_KEYS: set[str] = {
    "tile_forbidden",
    "tile_empty",
    "tile_hole",
    "tile_deepwater",
    "tile_wall",
}


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


from core.base.paths import game_data


def _levels_json_path() -> Path:
    """返回 levels.json 的查找路径。"""
    return game_data("levels.json")


_LEVELS_CACHE: Optional[list] = None
_LEVELS_MTIME: Optional[float] = None
_LEVELS_TILES_MAP: Optional[dict[str, list]] = None


def _load_levels() -> list:
    """加载并缓存 levels.json 内容。"""
    global _LEVELS_CACHE, _LEVELS_MTIME, _LEVELS_TILES_MAP

    p = _levels_json_path()
    if not p.exists():
        _LEVELS_CACHE = None
        _LEVELS_MTIME = None
        _LEVELS_TILES_MAP = None
        return []

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = None

    if _LEVELS_CACHE is not None and _LEVELS_MTIME is not None and mtime == _LEVELS_MTIME:
        return _LEVELS_CACHE

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        _LEVELS_CACHE = data
        _LEVELS_MTIME = mtime
        _LEVELS_TILES_MAP = None
        return data
    except Exception:
        return []


def _load_tiles_map() -> dict[str, list]:
    """构建并缓存 code -> tiles 的映射。"""
    global _LEVELS_TILES_MAP
    if _LEVELS_TILES_MAP is None:
        _LEVELS_TILES_MAP = {}
        for lv in _load_levels():
            code = lv.get("code")
            if code:
                _LEVELS_TILES_MAP[code] = lv.get("tiles", [])
    return _LEVELS_TILES_MAP


def load_stage_tiles(code: str) -> Optional[list]:
    """根据关卡 code 从 levels.json 加载 tiles 数据（含 heightType）。"""
    return _load_tiles_map().get(code)


def invalidate_levels_cache() -> None:
    """清空 levels.json 缓存，通常在文件被替换后调用。"""
    global _LEVELS_CACHE, _LEVELS_MTIME, _LEVELS_TILES_MAP
    _LEVELS_CACHE = None
    _LEVELS_MTIME = None
    _LEVELS_TILES_MAP = None


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


def _load_tiles_from_json(code: Optional[str] = None, name: Optional[str] = None) -> Optional[list]:
    """从本地 levels.json 加载指定关卡的 tiles 数据（含 heightType）。"""
    if code:
        tiles = load_stage_tiles(code)
        if tiles:
            return tiles
    for lv in _load_levels():
        if name and lv.get("name") == name:
            return lv.get("tiles")
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
        tiles: Optional[list] = None,
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

        # 加载 tiles 以获取 heightType；优先使用传入的 tiles
        if tiles is not None:
            self.tiles = tiles
        elif stage_code or stage_name:
            self.tiles = _load_tiles_from_json(stage_code, stage_name)
        else:
            self.tiles = None

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

    def _get_tile_height_type(self, row: int, col: int) -> int:
        """返回指定格子的 heightType（未加载 tiles 时视为 0）。"""
        if self.tiles and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            try:
                return self.tiles[row][col].get("heightType", 0)
            except Exception:
                pass
        return 0

    def _get_tile_height(self, row: int, col: int) -> float:
        """返回指定格子的世界坐标 z 值（heightType * -0.4）。

        与 Arknights-Tile-Pos 参考实现保持一致。若未加载 tiles 则按平地处理。
        """
        return self._get_tile_height_type(row, col) * -0.4

    def _is_tile_deployable(self, row: int, col: int) -> bool:
        """判断指定格子是否可部署。

        优先使用 buildableType（0 表示不可部署）；若缺失则用 tileKey 兜底。
        """
        if self.tiles and 0 <= row < self.grid_rows and 0 <= col < self.grid_cols:
            try:
                tile = self.tiles[row][col]
                buildable = tile.get("buildableType")
                if buildable is not None:
                    return buildable != 0
                tile_key = tile.get("tileKey")
                if tile_key and tile_key != "unknown":
                    return tile_key not in _NON_DEPLOYABLE_TILE_KEYS
            except Exception:
                pass
        return True

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
        wz = self._get_tile_height(row, col)

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

    def get_side_deploy_offset_vector(
        self, offset_px_base: float = 20.0, base_h: int = 1600
    ) -> Tuple[float, float]:
        """返回 side 视角下部署落点相对 tile 中心的屏幕偏移向量。

        side 视角中角色实际落点相对 tile 中心偏下一行。本方法返回从 tile 中心指向
        实际落点的向量，可直接加到 tile 中心坐标上得到实际落点；长度按
        ``offset_px_base`` 像素（按 ``base_h`` 基准高度缩放）。
        """
        matrix = self._get_transform_matrix(side=True)
        origin = np.dot(matrix, np.array([0.0, 0.0, 0.0, 1.0]))
        y_plus = np.dot(matrix, np.array([0.0, 1.0, 0.0, 1.0]))

        def _to_screen(p):
            return (
                (1 + p[0] / p[3]) / 2 * self.screen_width,
                (1 - p[1] / p[3]) / 2 * self.screen_height,
            )

        sx0, sy0 = _to_screen(origin)
        sx1, sy1 = _to_screen(y_plus)
        # world y+1 指向“上一行”（tile 中心方向），实际落点在其反方向，所以取负。
        dx = sx0 - sx1
        dy = sy0 - sy1
        length = math.hypot(dx, dy)
        scale = self.screen_height / base_h
        if length == 0:
            return 0.0, offset_px_base * scale
        return dx / length * offset_px_base * scale, dy / length * offset_px_base * scale

    def get_tile_polygon(
        self, row: int, col: int, side: bool = False
    ) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
        """返回指定格子在屏幕上的投影四边形（左上/右上/右下/左下顺序不严格）。

        基于格子在世界坐标中占据 [col-0.5, col+0.5] x [row-0.5, row+0.5]
        的矩形，投影四个角到屏幕。已考虑 heightType 带来的 z 偏移。
        """
        matrix = self._get_transform_matrix(side)
        wz = self._get_tile_height(row, col)

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
            px, py, _, pw = np.dot(matrix, np.array([sx_world, sy_world, wz, 1]))
            sx = (1 + px / pw) / 2 * self.screen_width
            sy = (1 - py / pw) / 2 * self.screen_height
            screen_corners.append((int(sx), int(sy)))
        return tuple(screen_corners)  # type: ignore[return-value]

    def hit_test(
        self, x: int, y: int, side: bool = False
    ) -> Optional[Tuple[int, int]]:
        """判断屏幕点 (x, y) 落在哪个格子的投影四边形内部。

        side 视角下多个格子投影可能重叠：优先选择可部署的高台格，
        其次可部署的地面格；不可部署的格子（如 tile_forbidden）仅作为视觉背景，
        不会遮挡背后的可部署格子。同优先级取中心点最近者。
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
                    is_deployable = self._is_tile_deployable(r, c)
                    height_type = self._get_tile_height_type(r, c)
                    # 可部署 > 不可部署；可部署中高台 > 地面；同优先级取中心最近
                    candidates.append((-int(is_deployable), -height_type, d, r, c))
        if candidates:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            return candidates[0][3], candidates[0][4]
        return None
