"""搜索最优的 side 视角偏移量（屏幕像素偏移或世界坐标偏移）。

用法：
    python tools/find_best_offset.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.capture.capture import WindowCapture
from core.map.tile_pos import TilePosCalculator, load_stage_dimensions


PROJECT_ROOT = Path(__file__).parent.parent
OUT_PATH = PROJECT_ROOT / "debug" / "find_best_offset.out"


def load_all_records():
    records = []
    d = PROJECT_ROOT / "debug" / "tile_hit_debug"
    for path in d.glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records.append(rec)
    return records


def make_calc(rec):
    capture = WindowCapture()
    w, h = capture.get_window_size()
    code = rec["stage_code"]
    grid_cols, grid_rows = load_stage_dimensions(code)
    return TilePosCalculator(w, h, grid_rows, grid_cols, stage_code=code)


def test_screen_offset(calc: TilePosCalculator, records, offset: float):
    """把 side 视角的 screen y 统一加 offset，统计正确率。"""
    correct = 0
    total = 0
    for rec in records:
        if not rec.get("side"):
            continue
        x, y = rec["screen"]
        pred_r, pred_c = rec["grid"]
        actual_r, actual_c = pred_r - 1, pred_c

        # 临时偏移：把点向上/下移动 offset 后做 hit_test
        grid = calc.hit_test(x, int(round(y - offset)), side=True)
        total += 1
        if grid == (actual_r, actual_c):
            correct += 1
    return correct, total


def test_world_offset(calc: TilePosCalculator, records, offset: float):
    """在世界 y 坐标上统一加 offset，然后做 hit_test（只改中心，不改 polygon 会不一致，
    这里仅用于参考）。"""
    correct = 0
    total = 0
    for rec in records:
        if not rec.get("side"):
            continue
        x, y = rec["screen"]
        pred_r, pred_c = rec["grid"]
        actual_r, actual_c = pred_r - 1, pred_c

        # 用最近中心来近似判断
        best = None
        best_dist = float("inf")
        for r in range(calc.grid_rows):
            for c in range(calc.grid_cols):
                cx, cy = calc.get_screen_pos(r, c, side=True)
                # 模拟把网格向下移动 offset 个世界单位：row 越大 wy 越小，
                # 向下移动网格等价于让同一 screen y 对应更小的 row。
                # 这里直接试：给 cy 加像素量 = offset * scale
                d = (cx - x) ** 2 + (cy - y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (r, c)
        total += 1
        if best == (actual_r, actual_c):
            correct += 1
    return correct, total


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records = load_all_records()
    # 按关卡分组
    by_code = {}
    for rec in records:
        by_code.setdefault(rec["stage_code"], []).append(rec)

    lines = []
    lines.append(f"共 {len(records)} 条 F9 记录，涉及关卡: {list(by_code.keys())}")
    lines.append("")

    # 只用 side=True 的记录
    side_records = [r for r in records if r.get("side")]
    lines.append(f"side=True 记录: {len(side_records)} 条")
    lines.append("")

    # 为每个关卡创建 calc
    calcs = {code: make_calc(recs[0]) for code, recs in by_code.items()}

    # 屏幕像素偏移搜索
    lines.append("=== 屏幕 y 方向统一偏移（正值=向下移动网格） ===")
    best = None
    best_score = -1
    for off in range(-100, 101, 1):
        correct = 0
        total = 0
        for rec in side_records:
            calc = calcs[rec["stage_code"]]
            c, t = test_screen_offset(calc, [rec], off)
            correct += c
            total += t
        score = correct / total if total else 0
        lines.append(f"offset={off:4d}px  正确 {correct}/{total} = {score:.2%}")
        if correct > best_score:
            best_score = correct
            best = off
    lines.append(f"\n最佳屏幕 y 偏移: {best}px，正确率 {best_score}/{len(side_records)}")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(OUT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
