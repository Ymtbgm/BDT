"""采集费用/数量数字训练样本。

用法：
    python tools/capture_digit_dataset.py --total 12
    python tools/capture_digit_dataset.py --total 14 --window-title 明日方舟

操作：
    F9  : 截取当前部署栏所有 slot 的费用数字和数量角标
    F10 : 截取鼠标所在 slot 的费用/数量（用于定向补数据）
    ESC : 退出

提示：Windows 下 F9/F10 全局热键需要以管理员身份运行本脚本。

后续整理：
    把裁剪出的数字图按真实值分类到训练目录即可，例如：
        data/digit_train/cost/12/xxx.png
        data/digit_train/quantity/3/xxx.png
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import keyboard
import numpy as np
import win32gui
import win32con

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture.capture import WindowCapture
from core.vision.cost_recognition import preprocess_cost_image, preprocess_cost_image_inv
import core.base.constants as constants

# 数量角标默认 ROI（<=12 slot 动态估算）
_QUANTITY_Y_RATIO = 1535 / 1600
_QUANTITY_H_RATIO = 65 / 1600

# 部署栏整栏 ROI（仅用于保存参考图）
_BAR_TOP_RATIO = 1370 / 1600
_BAR_HEIGHT_RATIO = 230 / 1600


from core.base.paths import position_data


def _load_calibration_config(total_slots: int, name: str) -> Optional[dict]:
    """尝试加载已有的 13+ slot 标定配置。"""
    candidates = [
        position_data(f"{name}_config_total{total_slots}.json"),
        position_data(f"{name}_config.json"),
    ]
    for path in candidates:
        if path.exists():
            try:
                import json
                with path.open("r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return None


def _set_window_topmost(win_name: str) -> None:
    """将 OpenCV 控制窗口置顶，方便一边游戏一边按热键。"""
    try:
        hwnd = win32gui.FindWindow(None, win_name)
        if hwnd:
            win32gui.SetWindowPos(
                hwnd,
                win32con.HWND_TOPMOST,
                0,
                0,
                0,
                0,
                win32con.SWP_NOMOVE | win32con.SWP_NOSIZE,
            )
    except Exception as e:
        print(f"窗口置顶失败: {e}")


def _get_cost_roi(
    cap: WindowCapture,
    bar_index: int,
    total_slots: int,
) -> Tuple[int, int, int, int]:
    """返回费用数字 ROI (x, y, w, h) 的屏幕/客户区坐标。

    与 core/cost_recognition.py 保持一致：ROI 左边缘对齐 slot 中心线，
    宽度固定 53px，高度覆盖费用数字条。
    """
    w, h = cap.get_window_size()
    left = cap.monitor.get("left", 0)
    top = cap.monitor.get("top", 0)

    ratios = constants.DEPLOY_BAR_COST_ROI_RATIOS
    y = int(round(h * ratios[1])) + top
    rh = int(round(h * ratios[3]))

    cell_w = w / 12 if total_slots <= 12 else w / total_slots
    cx = w - cell_w * (bar_index + 0.5)
    rw = 53
    x = int(round(cx)) + left
    return x, y, rw, rh


def _get_quantity_roi(
    cap: WindowCapture,
    bar_index: int,
    total_slots: int,
) -> Tuple[int, int, int, int]:
    """返回数量角标 ROI (x, y, w, h) 的屏幕/客户区坐标。

    与 core/resolver.py 的动态估算保持一致：ROI 左边缘对齐 slot 中心线，
    宽度为半个格子宽，覆盖右下角数量角标区域。
    """
    w, h = cap.get_window_size()
    left = cap.monitor.get("left", 0)
    top = cap.monitor.get("top", 0)

    cell_w = w / 12 if total_slots <= 12 else w / total_slots
    cx = w - cell_w * (bar_index + 0.5)
    x = int(round(cx)) + left
    rw = int(round(cell_w / 2))
    rh = int(round(h * _QUANTITY_H_RATIO))
    y = int(round(h * _QUANTITY_Y_RATIO)) + top
    return x, y, rw, rh


def _capture_full_bar(cap: WindowCapture) -> np.ndarray:
    w, h = cap.get_window_size()
    full = cap.capture()
    y = int(h * _BAR_TOP_RATIO)
    roi_h = int(h * _BAR_HEIGHT_RATIO)
    return full[y : y + roi_h, :, :]


def _imwrite_unicode(path: Path, img: np.ndarray) -> None:
    """兼容中文路径的 cv2.imwrite 封装。

    OpenCV 的 cv2.imwrite 在 Windows 上对非 ASCII 路径支持不佳，
    使用 cv2.imencode + tofile 绕过该限制。
    """
    suffix = path.suffix.lower()
    ext = ".png" if suffix in (".png", ".jpg", ".jpeg", ".bmp") else suffix
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise RuntimeError(f"cv2.imencode 失败: {path}")
    buf.tofile(str(path))


def _save_crops(
    img: np.ndarray,
    fixed: np.ndarray,
    inv: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    _imwrite_unicode(out_dir / f"{prefix}_raw.png", img)
    _imwrite_unicode(out_dir / f"{prefix}_fixed.png", fixed)
    _imwrite_unicode(out_dir / f"{prefix}_inv.png", inv)


def _get_mouse_slot(cap: WindowCapture, total_slots: int) -> int:
    """根据鼠标在窗口客户区的 x 位置推断最近 slot 索引。"""
    w, _ = cap.get_window_size()
    hwnd = cap._find_hwnd()
    cx_screen, _ = win32gui.GetCursorPos()
    cx_client, _ = win32gui.ScreenToClient(hwnd, (cx_screen, 0))
    cx_ratio = cx_client / w

    cell_w = 1 / 12 if total_slots <= 12 else 1 / total_slots
    # slot 中心比例: 1 - (i + 0.5) * cell_w
    best_i = 0
    best_dist = float("inf")
    for i in range(total_slots):
        center_ratio = 1.0 - (i + 0.5) * cell_w
        dist = abs(cx_ratio - center_ratio)
        if dist < best_dist:
            best_dist = dist
            best_i = i
    return best_i


def _debug_info(img: np.ndarray, label: str) -> str:
    if img is None or img.size == 0:
        return f"{label}: 空图像"
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if img.shape[2] == 4 else img
    return f"{label}: shape={img.shape}, mean={gray.mean():.1f}, min={gray.min()}, max={gray.max()}"


def _capture_slot(
    cap: WindowCapture,
    bar_index: int,
    total_slots: int,
    session_dir: Path,
    capture_idx: int,
    preview: bool = False,
) -> None:
    prefix = f"cap{capture_idx:04d}_slot_{bar_index:02d}"
    # 费用
    x, y, w, h = _get_cost_roi(cap, bar_index, total_slots)
    print(f"  slot[{bar_index}] cost_roi=({x},{y},{w},{h})")
    try:
        cost_img = cap.capture_roi(x, y, w, h)
        print(f"    {_debug_info(cost_img, 'cost_raw')}")
        cost_fixed = preprocess_cost_image(cost_img)
        cost_inv = preprocess_cost_image_inv(cost_img)
        _save_crops(cost_img, cost_fixed, cost_inv, session_dir / "cost", prefix)
    except Exception as e:
        print(f"  slot[{bar_index}] 费用截取失败: {e}")
        return

    # 数量
    x, y, w, h = _get_quantity_roi(cap, bar_index, total_slots)
    print(f"  slot[{bar_index}] quantity_roi=({x},{y},{w},{h})")
    try:
        qty_img = cap.capture_roi(x, y, w, h)
        print(f"    {_debug_info(qty_img, 'quantity_raw')}")
        qty_fixed = preprocess_cost_image(qty_img)
        qty_inv = preprocess_cost_image_inv(qty_img)
        _save_crops(qty_img, qty_fixed, qty_inv, session_dir / "quantity", prefix)
    except Exception as e:
        print(f"  slot[{bar_index}] 数量截取失败: {e}")

    if preview:
        # 预处理图被放大过，统一缩放到同一高度再拼接
        target_h = cost_fixed.shape[0]
        cost_raw_bgr = cv2.cvtColor(cost_img, cv2.COLOR_BGRA2BGR) if cost_img.shape[2] == 4 else cost_img
        qty_raw_bgr = cv2.cvtColor(qty_img, cv2.COLOR_BGRA2BGR) if qty_img.shape[2] == 4 else qty_img
        cost_raw_resized = cv2.resize(cost_raw_bgr, (cost_raw_bgr.shape[1] * target_h // cost_raw_bgr.shape[0], target_h))
        qty_raw_resized = cv2.resize(qty_raw_bgr, (qty_raw_bgr.shape[1] * target_h // qty_raw_bgr.shape[0], target_h))
        qty_fixed_resized = cv2.resize(qty_fixed, (qty_fixed.shape[1] * target_h // qty_fixed.shape[0], target_h))
        qty_inv_resized = cv2.resize(qty_inv, (qty_inv.shape[1] * target_h // qty_inv.shape[0], target_h))
        canvas = np.hstack([
            cost_raw_resized,
            cost_fixed,
            cost_inv,
            qty_raw_resized,
            qty_fixed_resized,
            qty_inv_resized,
        ])
        cv2.imshow("last_capture", canvas)
        cv2.waitKey(1)


def _capture_all_slots(
    cap: WindowCapture,
    total_slots: int,
    session_dir: Path,
    start_idx: int,
) -> int:
    print(f"正在截取 {total_slots} 个 slot...")
    full_bar = _capture_full_bar(cap)
    _imwrite_unicode(session_dir / f"full_bar_{start_idx:04d}.png", full_bar)

    for i in range(total_slots):
        _capture_slot(cap, i, total_slots, session_dir, start_idx)
    print(f"已保存到: {session_dir} (cap{start_idx:04d})")
    return start_idx + 1


def main():
    parser = argparse.ArgumentParser(description="采集费用/数量数字训练样本")
    parser.add_argument(
        "--total",
        "-t",
        type=int,
        required=True,
        help="当前部署栏总 slot 数（干员+道具+召唤物）",
    )
    parser.add_argument(
        "--window-title",
        type=str,
        default="明日方舟",
        help="游戏窗口标题，默认 明日方舟",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "debug" / "digit_dataset",
        help="输出根目录",
    )
    args = parser.parse_args()

    cap = WindowCapture(window_title=args.window_title)
    session_dir = args.out_dir / f"session_{time.strftime('%Y%m%d_%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print(f"输出目录: {session_dir}")
    print(f"总 slot 数: {args.total}")
    print("操作说明:")
    print("  F9  - 截取当前所有 slot 的费用和数量")
    print("  F10 - 截取鼠标所在 slot 的费用和数量")
    print("  ESC - 退出")
    print("提示: 请确保本程序以管理员运行，否则 F9/F10 全局热键可能无法响应。")

    # 创建置顶 OpenCV 控制窗口，方便查看状态；按键仍走 keyboard 全局热键
    win_name = "capture_digit_dataset (F9=全部 F10=单槽 ESC=退出)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 520, 140)
    hint = np.zeros((140, 520, 3), dtype=np.uint8)
    cv2.putText(hint, "F9: capture all", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(hint, "F10: capture mouse slot", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(hint, "ESC: quit", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.imshow(win_name, hint)
    _set_window_topmost(win_name)

    capture_idx = 0
    while True:
        if keyboard.is_pressed("f9"):
            print("[F9] 截取所有 slot")
            capture_idx = _capture_all_slots(cap, args.total, session_dir, capture_idx)
            time.sleep(0.5)
        elif keyboard.is_pressed("f10"):
            slot = _get_mouse_slot(cap, args.total)
            print(f"[F10] 截取鼠标所在 slot[{slot}]")
            _capture_slot(cap, slot, args.total, session_dir, capture_idx, preview=True)
            capture_idx += 1
            time.sleep(0.3)
        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        cv2.waitKey(1)
        time.sleep(0.05)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
