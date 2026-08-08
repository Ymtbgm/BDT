"""在部署栏整栏截图上标出指定 slot 的截取区域，用于检查 estimate_total 几何是否正确。"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import core.base.constants as constants
from core.capture.capture import WindowCapture

# 与 core/resolver.py 保持一致
_BAR_CAPTURE_TOP_RATIO = 1370 / 1600
_BAR_CAPTURE_HEIGHT_RATIO = 230 / 1600
_BAR_AVATAR_SIZE_RATIO = 120 / 1600
_BAR_AVATAR_Y_OFFSET_RATIO = -10 / 120
_BAR_CENTER_Y_RATIO = 1500 / 1600

_QUANTITY_ROI_Y_RATIO = 1560 / 1600
_QUANTITY_ROI_H_RATIO = 40 / 1600

_COLORS = {
    "avatar": (0, 255, 0),      # 绿色
    "cost": (0, 0, 255),        # 红色
    "quantity": (255, 0, 0),    # 蓝色
    "grid": (128, 128, 128),    # 灰色
}


def recover_window_size(bar_img: np.ndarray) -> tuple[int, int]:
    h_bar, w_bar = bar_img.shape[:2]
    window_width = w_bar
    window_height = int(round(h_bar / _BAR_CAPTURE_HEIGHT_RATIO))
    return window_width, window_height


def cell_width(window_width: int, total_slots: int) -> float:
    return window_width / 12 if total_slots <= 12 else window_width / total_slots


def slot_center_x(window_width: int, cell_w: float, bar_index: int) -> float:
    return window_width - cell_w * (bar_index + 0.5)


def avatar_roi(window_width: int, window_height: int, bar_index: int, total_slots: int) -> tuple[int, int, int, int]:
    cell_w = cell_width(window_width, total_slots)
    cx = slot_center_x(window_width, cell_w, bar_index)
    bar_top = window_height * _BAR_CAPTURE_TOP_RATIO
    bar_center_y = window_height * _BAR_CENTER_Y_RATIO
    cy_rel = bar_center_y - bar_top
    avatar_size = window_height * _BAR_AVATAR_SIZE_RATIO
    y_offset = avatar_size * _BAR_AVATAR_Y_OFFSET_RATIO
    crop_cx = int(round(cx))
    crop_cy = int(round(cy_rel + y_offset))
    crop_size = int(round(avatar_size))
    x1 = max(0, crop_cx - crop_size // 2)
    y1 = max(0, crop_cy - crop_size // 2)
    x2 = x1 + crop_size
    y2 = y1 + crop_size
    return x1, y1, x2, y2


def cost_roi(window_width: int, window_height: int, bar_index: int, total_slots: int) -> tuple[int, int, int, int]:
    cell_w = cell_width(window_width, total_slots)
    cx = slot_center_x(window_width, cell_w, bar_index)
    bar_top = window_height * _BAR_CAPTURE_TOP_RATIO
    cost_y_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[1]
    cost_h_window = window_height * constants.DEPLOY_BAR_COST_ROI_RATIOS[3]
    y1 = int(round(cost_y_window - bar_top))
    y2 = int(round(y1 + cost_h_window))
    x1 = int(round(cx))
    x2 = x1 + 53
    return x1, y1, x2, y2


def quantity_roi(window_width: int, window_height: int, bar_index: int, total_slots: int) -> tuple[int, int, int, int]:
    cell_w = cell_width(window_width, total_slots)
    cx = slot_center_x(window_width, cell_w, bar_index)
    bar_top = window_height * _BAR_CAPTURE_TOP_RATIO
    x1 = int(round(cx))
    x2 = x1 + int(round(cell_w / 2))
    y1 = int(round(window_height * _QUANTITY_ROI_Y_RATIO - bar_top))
    y2 = y1 + int(round(window_height * _QUANTITY_ROI_H_RATIO))
    return x1, y1, x2, y2


def draw_roi(
    canvas: np.ndarray,
    roi: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    h, w = canvas.shape[:2]
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w, x2))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return canvas
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        canvas,
        label,
        (x1 + 2, max(y1 - 4, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return canvas


def draw_all_slot_centers(canvas: np.ndarray, window_width: int, window_height: int, total_slots: int):
    cell_w = cell_width(window_width, total_slots)
    bar_top = window_height * _BAR_CAPTURE_TOP_RATIO
    y_top = 0
    y_bottom = canvas.shape[0]
    for i in range(total_slots):
        cx = int(round(slot_center_x(window_width, cell_w, i)))
        cv2.line(canvas, (cx, y_top), (cx, y_bottom), _COLORS["grid"], 1)
        cv2.putText(
            canvas,
            f"{i}",
            (cx - 5, 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            _COLORS["grid"],
            1,
            cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="标记部署栏整栏截图中某个 slot 的截取区域")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", "-i", type=Path, help="整栏截图路径")
    source.add_argument("--live", "-l", action="store_true", help="直接截取当前游戏窗口部署区")
    parser.add_argument("--window-title", type=str, default="明日方舟", help="游戏窗口标题，默认 明日方舟")
    parser.add_argument("--slot", "-s", type=int, required=True, help="要标记的 slot 索引（从右往左 0 开始）")
    parser.add_argument("--total", "-t", type=int, required=True, help="假设的部署区总 slot 数")
    parser.add_argument("--output", "-o", type=Path, help="输出图片路径，默认 debug/visualize_deploy_bar_roi/<time>.png")
    parser.add_argument("--no-grid", action="store_true", help="不绘制所有 slot 中心线")
    args = parser.parse_args()

    if args.image:
        if not args.image.exists():
            print(f"图片不存在: {args.image}")
            raise SystemExit(1)
        bar_img = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        if bar_img is None:
            print(f"读取图片失败: {args.image}")
            raise SystemExit(1)
        window_width, window_height = recover_window_size(bar_img)
        print(f"从文件加载: {args.image}")
    else:
        cap = WindowCapture(window_title=args.window_title)
        window_width, window_height = cap.get_window_size()
        full_img = cap.capture()  # 优先 PrintWindow，支持后台/遮挡窗口
        x = 0
        y = int(window_height * _BAR_CAPTURE_TOP_RATIO)
        roi_w = window_width
        roi_h = int(window_height * _BAR_CAPTURE_HEIGHT_RATIO)
        bar_img = full_img[y : y + roi_h, x : x + roi_w]
        print(f"已截取游戏窗口部署区 (窗口内相对: {x},{y} {roi_w}x{roi_h})")

    print(f"窗口尺寸推断/获取: {window_width}x{window_height}")
    print(f"截图尺寸: {bar_img.shape[1]}x{bar_img.shape[0]}")

    canvas = bar_img.copy()
    if not args.no_grid:
        draw_all_slot_centers(canvas, window_width, window_height, args.total)

    avatar = avatar_roi(window_width, window_height, args.slot, args.total)
    cost = cost_roi(window_width, window_height, args.slot, args.total)
    quantity = quantity_roi(window_width, window_height, args.slot, args.total)

    print(f"slot[{args.slot}] @ total={args.total}")
    print(f"  头像 ROI: {avatar}")
    print(f"  费用 ROI: {cost}")
    print(f"  数量 ROI: {quantity}")

    canvas = draw_roi(canvas, avatar, _COLORS["avatar"], f"avatar[{args.slot}]")
    canvas = draw_roi(canvas, cost, _COLORS["cost"], f"cost[{args.slot}]")
    canvas = draw_roi(canvas, quantity, _COLORS["quantity"], f"qty[{args.slot}]")

    if args.output is None:
        out_dir = Path("debug") / "visualize_deploy_bar_roi"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / f"{int(time.time() * 1000)}_slot{args.slot}_total{args.total}.png"

    cv2.imwrite(str(args.output), canvas)
    print(f"已保存: {args.output}")


if __name__ == "__main__":
    main()
