import sys
import numpy as np
import math

# 参考项目的 Calc 类（精简版）
class RefCalc:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.ratio = height / width
        self.matrix_p = np.array([
            [self.ratio / math.tan(math.pi * 20 / 180), 0, 0, 0],
            [0, 1 / math.tan(math.pi * 20 / 180), 0, 0],
            [0, 0, -(1000 + 0.3) / (1000 - 0.3), -(1000 * 0.3 * 2) / (1000 - 0.3)],
            [0, 0, -1, 0],
        ])

    def adapter(self):
        fromRatio = 9 / 16
        toRatio = 3 / 4
        if self.ratio < fromRatio - 0.00001:
            return 0, 0
        t = (self.ratio - fromRatio) / (toRatio - fromRatio)
        return -1.4 * t, -2.8 * t

    def run(self, h, w, side=False):
        view0 = [0.0, -4.81, -7.76]
        view1 = [0.5975098586953793, -5.31, -8.642108163374733]
        x, y, z = view1 if side else view0
        adapter_y, adapter_z = self.adapter()
        y += adapter_y
        z += adapter_z
        raw = np.array([
            [1, 0, 0, -x],
            [0, 1, 0, -y],
            [0, 0, 1, -z],
            [0, 0, 0, 1],
        ])
        matrix_x = np.array([
            [1, 0, 0, 0],
            [0, math.cos(math.pi * 30 / 180), -math.sin(math.pi * 30 / 180), 0],
            [0, -math.sin(math.pi * 30 / 180), -math.cos(math.pi * 30 / 180), 0],
            [0, 0, 0, 1],
        ])
        matrix_y = np.array([
            [math.cos(math.pi * 10 / 180), 0, math.sin(math.pi * 10 / 180), 0],
            [0, 1, 0, 0],
            [-math.sin(math.pi * 10 / 180), 0, math.cos(math.pi * 10 / 180), 0],
            [0, 0, 0, 1],
        ])
        if side:
            matrix = np.dot(matrix_x, matrix_y)
            matrix = np.dot(matrix, raw)
        else:
            matrix = np.dot(matrix_x, raw)
        matrix = np.dot(self.matrix_p, matrix)
        result = []
        for y in range(h):
            tmp = []
            for x in range(w):
                p_x, p_y, p_z, p_w = np.dot(matrix,
                    np.array([(x - (w - 1) / 2), ((h - 1) / 2) - y, 0.0 * -0.4, 1]))
                p_x = (1 + p_x / p_w) / 2
                p_y = (1 + p_y / p_w) / 2
                center = int(p_x * self.width), int((1 - p_y) * self.height)
                tmp.append(center)
            result.append(tmp)
        return result


# 我们的 TilePosCalculator
class OurCalc:
    def __init__(self, screen_width, screen_height, grid_rows, grid_cols):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.ratio = screen_height / screen_width
        self.view_normal = (0.0, -4.81, -7.76)
        self.view_side = (0.5975098586953793, -5.31, -8.642108163374733)
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

    def _adapter(self):
        from_ratio = 9 / 16
        to_ratio = 3 / 4
        if self.ratio < from_ratio - 0.00001:
            return 0.0, 0.0
        t = (self.ratio - from_ratio) / (to_ratio - from_ratio)
        return -1.4 * t, -2.8 * t

    def _get_transform_matrix(self, side=False):
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

    def get_screen_pos(self, row, col, side=False):
        h, w = self.grid_rows, self.grid_cols
        wx = col - (w - 1) / 2
        wy = (h - 1) / 2 - row
        wz = 0.0
        matrix = self._get_transform_matrix(side)
        px, py, _, pw = np.dot(matrix, np.array([wx, wy, wz, 1]))
        sx = (1 + px / pw) / 2 * self.screen_width
        sy = (1 - py / pw) / 2 * self.screen_height
        return int(sx), int(sy)


if __name__ == "__main__":
    w, h = 2560, 1600
    rows, cols = 8, 11
    side = False

    ref = RefCalc(w, h)
    our = OurCalc(w, h, rows, cols)

    ref_result = ref.run(rows, cols, side)

    print(f"对比 {rows}x{cols} 地图, side={side}, 屏幕 {w}x{h}")
    print(f"{'格子':<12} {'参考项目':<20} {'我们的代码':<20} {'差异':<15}")
    print("-" * 70)

    all_match = True
    for r in range(rows):
        for c in range(cols):
            ref_x, ref_y = ref_result[r][c]
            our_x, our_y = our.get_screen_pos(r, c, side)
            dx = ref_x - our_x
            dy = ref_y - our_y
            if dx != 0 or dy != 0:
                all_match = False
                print(f"({r},{c}){'':<6} ({ref_x},{ref_y}){'':<8} ({our_x},{our_y}){'':<8} dx={dx}, dy={dy}")

    if all_match:
        print("所有坐标完全一致！")
    else:
        print("\n发现差异，需要检查代码逻辑。")
