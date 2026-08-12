"""根据 F9 记录分析 side 视角下行偏移量。

用法：
    python tools/analyze_tile_offset.py

读取 debug/tile_hit_debug/ 下最新的 AS-9 / TO-9 记录，
对每一条预测为 (r, c)、实际为 (r-1, c) 的点，
计算需要把 side 视角网格向下调整多少像素/世界坐标，
才能让 hit_test 返回实际格子。
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capture.capture import WindowCapture
from core.map.tile_pos import TilePosCalculator, load_stage_dimensions


PROJECT_ROOT = Path(__file__).parent.parent
OUT_PATH = PROJECT_ROOT / "debug" / "analyze_tile_offset.out"


def log(msg: str):
    print(msg)
    with open(OUT_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_logs(stage_code: str):
    d = PROJECT_ROOT / "debug" / "tile_hit_debug"
    files = sorted(d.glob(f"{stage_code}_*.txt"))
    return files[-1] if files else None


def load_records(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def analyze_stage(stage_code: str, capture: WindowCapture):
    path = find_logs(stage_code)
    if not path:
        log(f"未找到 {stage_code} 的 F9 记录")
        return

    records = load_records(path)
    w, h = capture.get_window_size()
    dims = load_stage_dimensions(stage_code)
    if not dims:
        log(f"未找到 {stage_code} 的尺寸")
        return
    grid_cols, grid_rows = dims
    calc = TilePosCalculator(w, h, grid_rows, grid_cols, stage_code=stage_code)

    log(f"\n=== {stage_code} ({grid_rows}x{grid_cols}, window {w}x{h}) ===")
    log(f"记录文件: {path}")

    screen_offsets = []

    for rec in records:
        x, y = rec["screen"]
        pred_r, pred_c = rec["grid"]
        actual_r, actual_c = pred_r - 1, pred_c

        # 当前 hit_test 确实是预测格子
        now = calc.hit_test(x, y, side=True)
        if now != (pred_r, pred_c):
            log(f"\npoint=({x},{y}) 当前 hit_test={now}，与记录的预测 {pred_r, pred_c} 不一致，跳过")
            continue

        # 在屏幕 y 方向二分搜索：加多少像素偏移能让 hit_test 变成实际格子
        # 正值 = 向下移动；负值 = 向上移动
        lo, hi = -200.0, 200.0
        for _ in range(50):
            mid = (lo + hi) / 2
            grid = calc.hit_test(x, int(round(y + mid)), side=True)
            if grid == (actual_r, actual_c):
                hi = mid
            else:
                lo = mid
        screen_offset = (lo + hi) / 2
        screen_offsets.append(screen_offset)

        poly_pred = calc.get_tile_polygon(pred_r, pred_c, side=True)
        poly_actual = calc.get_tile_polygon(actual_r, actual_c, side=True)

        # 计算点到实际多边形各边的距离，以及到预测多边形各边的距离
        import cv2
        pt = (float(x), float(y))
        dist_actual = cv2.pointPolygonTest(
            np.array(poly_actual, dtype=np.float32), pt, True
        )
        dist_pred = cv2.pointPolygonTest(
            np.array(poly_pred, dtype=np.float32), pt, True
        )

        log(f"\npoint=({x}, {y})  pred=({pred_r},{pred_c}) actual=({actual_r},{actual_c})")
        log(f"  预测格子多边形: {poly_pred}")
        log(f"  实际格子多边形: {poly_actual}")
        log(f"  点到实际多边形边界距离: {dist_actual:.1f}px（正=内部）")
        log(f"  点到预测多边形边界距离: {dist_pred:.1f}px（正=内部）")
        log(f"  需要把点在屏幕 y 方向移动 {screen_offset:.2f}px 才能命中实际格子")

    if screen_offsets:
        avg = sum(screen_offsets) / len(screen_offsets)
        log(f"\n{stage_code} 屏幕 y 方向平均偏移: {avg:.2f}px")
        log(f"  各点偏移: {[round(v, 2) for v in screen_offsets]}")
        # 负值表示需要向上移动；正值表示需要向下移动
        # 如果平均值是负的，说明我们的网格整体偏下，需要向上修正
        if avg < 0:
            log(f"  结论：side 视角网格需要向上修正约 {abs(avg):.2f}px")
        else:
            log(f"  结论：side 视角网格需要向下修正约 {avg:.2f}px")


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("", encoding="utf-8")
    capture = WindowCapture()
    for code in ("AS-9", "TO-9"):
        analyze_stage(code, capture)


if __name__ == "__main__":
    main()
