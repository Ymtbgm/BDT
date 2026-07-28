"""标定 13+ slot 时部署栏数量角标的 ROI 中心。

核心观察：点击不同部署位时，整个部署栏会发生形变，因此数量角标位置不仅和
目标 slot 有关，还和当前被点中的 slot（active_slot）有关。理论上需要标定
slot^2 个位置。

使用方式：
    # 用已有截图一次性标定，默认保存到 data/quantity_roi_config_total13.json
    python tools/calibrate_quantity_roi.py -i debug/recordings/<session>/keyframes/deploy_bar_xxxx.png -t 13

    # 在游戏中分步标定：
    #   1. 工具提示 active_slot=0 时，进游戏点击 slot 0 进入部署状态；
    #   2. 按 F9 截取当前游戏部署区；
    #   3. 工具提示 target_slot=0..N-1，把鼠标依次移到对应数量角标上，按 F10 记录；
    #   4. 重复 1-3 直到所有 active_slot 标定完成。
    python tools/calibrate_quantity_roi.py --live --step -t 13

默认每个 total 保存到独立文件，避免互相覆盖：
    data/quantity_roi_config_total13.json
    data/quantity_roi_config_total14.json
    data/quantity_roi_config_total15.json

如需合并到一个文件，可指定 -o data/quantity_roi_config.json。
保存前会自动给已有文件创建 .bak 备份。
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import keyboard
import numpy as np
import win32con
import win32gui

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture

# 与 core/resolver.py 保持一致
_BAR_CAPTURE_TOP_RATIO = 1370 / 1600
_BAR_CAPTURE_HEIGHT_RATIO = 230 / 1600
_QUANTITY_ROI_Y_RATIO = 1535 / 1600
_QUANTITY_ROI_H_RATIO = 65 / 1600

# level logo 和数量角标都在部署栏底部这一窄带内，检测时只搜这里
_LOGO_SEARCH_TOP_RATIO = 1460 / 1600  # 窗口比例
_LOGO_SEARCH_HEIGHT_RATIO = 140 / 1600

# 数量角标相对 level logo 的默认偏移（2560x1600 基准）
# 用户确认：数量在 logo 最左端左侧 45 像素
_LOGO_TO_QUANTITY_DX_BASE = -45
_LOGO_TO_QUANTITY_DY_BASE = 0
_BASE_WIDTH = 2560
_BASE_HEIGHT = 1600


def _logo_left_ratio(
    cx_ratio: float, scale: float, window_width: int
) -> float:
    """由 logo 中心和匹配尺度计算 logo 最左端的窗口 x 比例。"""
    half_w_px = (53 * scale) / 2
    return cx_ratio - half_w_px / window_width


def recover_window_size(bar_img: np.ndarray) -> Tuple[int, int]:
    h_bar, w_bar = bar_img.shape[:2]
    window_width = w_bar
    window_height = int(round(h_bar / _BAR_CAPTURE_HEIGHT_RATIO))
    return window_width, window_height


def base_quantity_size(window_width: int, window_height: int) -> Tuple[float, float]:
    """以 12 slot 时的动态 ROI 作为基准尺寸（宽=cell_w/2，高=数量条高度）。"""
    w = window_width / 24  # (window_width / 12) / 2
    h = window_height * _QUANTITY_ROI_H_RATIO
    return w, h


def to_window_ratios(
    pt: Tuple[int, int], window_width: int, window_height: int
) -> Tuple[float, float]:
    u, v = pt
    x_ratio = u / window_width
    y_ratio = v / window_height + _BAR_CAPTURE_TOP_RATIO
    return x_ratio, y_ratio


def from_window_ratios(
    cx_ratio: float, cy_ratio: float, window_width: int, window_height: int
) -> Tuple[int, int]:
    x = cx_ratio * window_width
    y = (cy_ratio - _BAR_CAPTURE_TOP_RATIO) * window_height
    return int(round(x)), int(round(y))


def dynamic_slot_center_x_ratio(slot_index: int, total_slots: int) -> float:
    return 1.0 - (slot_index + 0.5) / total_slots


def nearest_slot_index(x_ratio: float, total_slots: int) -> int:
    return min(
        range(total_slots),
        key=lambda i: abs(x_ratio - dynamic_slot_center_x_ratio(i, total_slots)),
    )


def draw_state(
    canvas: np.ndarray,
    centers: Dict[int, Dict[int, Tuple[float, float]]],
    half_w: float,
    half_h: float,
    window_width: int,
    window_height: int,
    total_slots: int,
    active_slot: int = -1,
    target_slot: int = -1,
    waiting_active: bool = False,
) -> np.ndarray:
    out = canvas.copy()
    h_bar, w_bar = out.shape[:2]

    # 绘制动态估算的 slot 中心线作为参考
    for i in range(total_slots):
        cx_ratio = dynamic_slot_center_x_ratio(i, total_slots)
        x = int(round(cx_ratio * window_width))
        if 0 <= x < w_bar:
            if i == active_slot:
                color = (0, 0, 255)  # 红色：当前 active slot
                thickness = 2
            elif i == target_slot and not waiting_active:
                color = (255, 0, 0)  # 蓝色：当前 target slot
                thickness = 2
            else:
                color = (64, 64, 64)
                thickness = 1
            cv2.line(out, (x, 0), (x, h_bar), color, thickness)
            cv2.putText(
                out,
                str(i),
                (max(0, x - 5), 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (128, 128, 128),
                1,
                cv2.LINE_AA,
            )

    # 绘制已标定的 ROI（当前 active 用绿色，其他 active 用青色）
    for a_slot, targets in sorted(centers.items()):
        for t_slot, (cx_ratio, cy_ratio) in sorted(targets.items()):
            cx, cy = from_window_ratios(cx_ratio, cy_ratio, window_width, window_height)
            x1 = int(round(cx - half_w))
            y1 = int(round(cy - half_h))
            x2 = int(round(cx + half_w))
            y2 = int(round(cy + half_h))
            x1 = max(0, min(w_bar - 1, x1))
            y1 = max(0, min(h_bar - 1, y1))
            x2 = max(x1 + 1, min(w_bar, x2))
            y2 = max(y1 + 1, min(h_bar, y2))
            color = (0, 255, 0) if a_slot == active_slot else (255, 255, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"a{a_slot}t{t_slot}"
            cv2.putText(
                out,
                label,
                (x1 + 2, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )

    # 操作提示
    if active_slot >= 0:
        if waiting_active:
            status = f"请进游戏点击 slot[{active_slot}] 进入部署状态，然后按 F9 截图"
        else:
            status = f"active={active_slot} target={target_slot}，把鼠标移到该角标上按 F10"
        hints = [
            "F9: 截取部署区  |  F10: 记录鼠标位置  |  y: 重固定 y  |  n: 跳过  |  z: 撤销  |  r: 重置  |  s: 保存  |  Esc: 退出",
            f"total={total_slots}  active={active_slot}/{total_slots}  target={target_slot}/{total_slots}  {status}",
        ]
    else:
        hints = [
            "Left click: mark slot center  |  z: undo  |  r: reset  |  s: save & quit  |  q/Esc: quit",
            f"total={total_slots}",
        ]
    for row, hint in enumerate(hints):
        y = h_bar - 8 - row * 18
        cv2.putText(
            out,
            hint,
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out


def load_config(path: Path) -> dict:
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"读取已有配置失败: {e}，将创建新配置")
    return {"version": "1.1", "base_size": {}, "calibrations": {}}


def save_config(
    path: Path,
    config: dict,
    total_slots: int,
    centers: Dict[int, Dict[int, Tuple[float, float]]],
    half_w_ratio: float,
    half_h_ratio: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存前自动备份已有配置，防止误覆盖
    if path.exists():
        backup_path = path.with_suffix(".json.bak")
        try:
            shutil.copy2(path, backup_path)
        except Exception as e:
            print(f"备份失败: {e}")

    rois_by_active: Dict[str, List[dict]] = {}
    for a_slot in sorted(centers.keys()):
        rois: List[dict] = []
        for t_slot in sorted(centers[a_slot].keys()):
            cx_ratio, cy_ratio = centers[a_slot][t_slot]
            rois.append(
                {
                    "slot": t_slot,
                    "cx_ratio": cx_ratio,
                    "cy_ratio": cy_ratio,
                    "x1_ratio": cx_ratio - half_w_ratio,
                    "y1_ratio": cy_ratio - half_h_ratio,
                    "x2_ratio": cx_ratio + half_w_ratio,
                    "y2_ratio": cy_ratio + half_h_ratio,
                }
            )
        rois_by_active[str(a_slot)] = rois

    config["base_size"] = {
        "w_ratio": half_w_ratio * 2,
        "h_ratio": half_h_ratio * 2,
    }
    config["calibrations"][str(total_slots)] = rois_by_active

    with path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    total_rois = sum(len(v) for v in rois_by_active.values())
    print(f"已保存标定结果: {path} (total={total_slots}, rois={total_rois})")


def capture_bar(cap: WindowCapture) -> Tuple[np.ndarray, int, int]:
    window_width, window_height = cap.get_window_size()
    full_img = cap.capture()
    x = 0
    y = int(window_height * _BAR_CAPTURE_TOP_RATIO)
    roi_w = window_width
    roi_h = int(window_height * _BAR_CAPTURE_HEIGHT_RATIO)
    bar_img = full_img[y : y + roi_h, x : x + roi_w]
    return bar_img, window_width, window_height


def get_mouse_window_ratio(cap: WindowCapture) -> Tuple[float, float]:
    """获取当前鼠标相对于游戏窗口客户区的比例坐标。"""
    hwnd = cap._find_hwnd()
    cx_screen, cy_screen = win32gui.GetCursorPos()
    cx_client, cy_client = win32gui.ScreenToClient(hwnd, (cx_screen, cy_screen))
    window_width, window_height = cap.get_window_size()
    return cx_client / window_width, cy_client / window_height


def set_window_topmost(win_name: str) -> None:
    """将 OpenCV 窗口置顶，方便一边游戏一边标定。"""
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


LEVEL_LOGO_PATH = ROOT / "core" / "resource" / "level_logo.png"
LEVEL_LOGO_WHITE_THRESHOLD = 200


def _to_binary(img: np.ndarray, threshold: int) -> np.ndarray:
    """非白色像素置黑，用于检测 level logo 等纯白标识。"""
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def detect_level_logos(
    bar_img: np.ndarray,
    total_slots: int,
    window_width: int,
    window_height: int,
    threshold: int = LEVEL_LOGO_WHITE_THRESHOLD,
) -> List[Tuple[float, float, float, float]]:
    """基于白色阈值 + 多尺度模板匹配检测部署栏底部窄带中的 level logo。

    返回窗口比例坐标列表 [(cx_ratio, cy_ratio, score, scale), ...]，按 slot 从右到左排序。
    """
    if not LEVEL_LOGO_PATH.exists():
        raise FileNotFoundError(f"未找到 level logo 模板: {LEVEL_LOGO_PATH}")

    tmpl = _imread_unicode(LEVEL_LOGO_PATH, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        raise ValueError(f"无法读取 level logo 模板: {LEVEL_LOGO_PATH}")

    # 只在数量条/Logo 所在底部窄带搜索，减少误检和计算量
    bar_h, bar_w = bar_img.shape[:2]
    search_top_ratio = (_LOGO_SEARCH_TOP_RATIO - _BAR_CAPTURE_TOP_RATIO) / _BAR_CAPTURE_HEIGHT_RATIO
    search_h_ratio = _LOGO_SEARCH_HEIGHT_RATIO / _BAR_CAPTURE_HEIGHT_RATIO
    search_top = int(bar_h * search_top_ratio)
    search_bottom = int(bar_h * (search_top_ratio + search_h_ratio))
    search_img = bar_img[search_top:search_bottom, :]

    binary_search = _to_binary(search_img, threshold)
    binary_tmpl = _to_binary(tmpl, threshold)

    # 以 1600h 下头像约 120px、模板高度 46px 为基准估算当前分辨率缩放
    scale_ref = window_height / 1600
    scales = [scale_ref * s for s in np.linspace(0.7, 1.3, 13)]

    detections: List[Tuple[float, float, float, float]] = []  # cx_search, cy_search, scale, score
    for scale in scales:
        resized = cv2.resize(binary_tmpl, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        rh, rw = resized.shape
        if rh > binary_search.shape[0] or rw > binary_search.shape[1]:
            continue
        result = cv2.matchTemplate(binary_search, resized, cv2.TM_CCOEFF_NORMED)
        loc = np.where(result >= 0.65)
        for pt in zip(*loc[::-1]):
            cx = pt[0] + rw / 2
            cy = pt[1] + rh / 2
            score = result[pt[1], pt[0]]
            detections.append((cx, cy, scale, score))

    if not detections:
        return []

    # 按分数排序后做非极大值抑制
    detections.sort(key=lambda x: x[3], reverse=True)
    tmpl_h, tmpl_w = binary_tmpl.shape
    base_size = max(tmpl_w, tmpl_h) * scale_ref
    min_dist = base_size * 0.6

    filtered: List[Tuple[float, float, float, float]] = []
    for cx_search, cy_search, scale, score in detections:
        too_close = False
        for fx, fy, fs, fscale in filtered:
            # 比较时用 bar 像素坐标
            fcx = fx * window_width
            fcy = (fy - _BAR_CAPTURE_TOP_RATIO) * window_height
            cx_bar = cx_search
            cy_bar = search_top + cy_search
            if abs(cx_bar - fcx) < min_dist and abs(cy_bar - fcy) < min_dist:
                too_close = True
                break
        if not too_close:
            cx_bar = cx_search
            cy_bar = search_top + cy_search
            x_ratio = cx_bar / window_width
            y_ratio = cy_bar / window_height + _BAR_CAPTURE_TOP_RATIO
            filtered.append((x_ratio, y_ratio, score, scale))

    # 按 x 从右到左排序
    filtered.sort(key=lambda p: p[0], reverse=True)
    return filtered


def _next_unassigned_target(
    start: int, total: int, active: int, centers: Dict[int, Dict[int, Tuple[float, float]]]
) -> int:
    t = start
    while t < total and t in centers.get(active, {}):
        t += 1
    return t


def _next_unassigned_active(
    start: int, total: int, centers: Dict[int, Dict[int, Tuple[float, float]]]
) -> int:
    a = start
    while a < total and len(centers.get(a, {})) == total:
        a += 1
    return a


def run_step_mode(
    cap: WindowCapture,
    args,
    config: dict,
) -> None:
    """分步热键标定：F9 截图，F10 记录鼠标位置。支持 slot^2 标定。"""
    bar_img, window_width, window_height = capture_bar(cap)
    base_w, base_h = base_quantity_size(window_width, window_height)
    half_w, half_h = base_w / 2, base_h / 2
    half_w_ratio = half_w / window_width
    half_h_ratio = half_h / window_height

    print(f"推断窗口尺寸: {window_width}x{window_height}")
    print(f"基准 ROI 尺寸: {base_w:.1f}x{base_h:.1f} (窗口比例 {half_w_ratio*2:.4f}x{half_h_ratio*2:.4f})")

    # centers[active_slot][target_slot] = (cx_ratio, cy_ratio)
    centers: Dict[int, Dict[int, Tuple[float, float]]] = {}
    history: List[Tuple[int, int]] = []
    fixed_y_by_active: Dict[int, float] = {}
    # 默认偏移：数量角标在 level logo 左侧（2560x1600 基准）
    logo_offset: Optional[Tuple[float, float]] = (
        _LOGO_TO_QUANTITY_DX_BASE / _BASE_WIDTH,
        _LOGO_TO_QUANTITY_DY_BASE / _BASE_HEIGHT,
    )
    candidate_boxes: Dict[int, List[Tuple[float, float]]] = {}  # active_slot -> 候选 quantity 框

    existing = config.get("calibrations", {}).get(str(args.total), {})
    if isinstance(existing, list):
        # 兼容旧格式：把列表当作 active_slot=0 的数据
        existing = {"0": existing}
    for a_slot_str, rois in existing.items():
        try:
            a_slot = int(a_slot_str)
        except ValueError:
            continue
        centers[a_slot] = {}
        for roi in rois:
            t_slot = roi.get("slot")
            if t_slot is not None and 0 <= t_slot < args.total:
                centers[a_slot][t_slot] = (roi["cx_ratio"], roi["cy_ratio"])

    active_slot = _next_unassigned_active(0, args.total, centers)
    target_slot = _next_unassigned_target(0, args.total, active_slot, centers)
    waiting_active = True

    display_bgr = (
        cv2.cvtColor(bar_img, cv2.COLOR_BGRA2BGR)
        if bar_img.shape[2] == 4
        else bar_img.copy()
    )
    redraw = True

    state = {
        "centers": centers,
        "history": history,
        "active_slot": active_slot,
        "target_slot": target_slot,
        "waiting_active": waiting_active,
        "fixed_y_by_active": fixed_y_by_active,
        "redraw": redraw,
        "window_width": window_width,
        "window_height": window_height,
        "total": args.total,
        "half_w": half_w,
        "half_h": half_h,
        "display_bgr": display_bgr,
        "logo_offset": logo_offset,
        "last_logo_positions": [],
        "candidate_boxes": candidate_boxes,
        "reset_confirm_until": 0.0,
    }

    win_name = f"calibrate_quantity_roi (total={args.total})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    # 分步模式下不需要鼠标回调，避免与游戏左键冲突
    cv2.setMouseCallback(win_name, lambda *a, **k: None)

    # 先显示一次并置顶，避免切回游戏后被遮挡
    canvas = draw_state(
        state["display_bgr"],
        state["centers"],
        state["half_w"],
        state["half_h"],
        state["window_width"],
        state["window_height"],
        state["total"],
        active_slot=state["active_slot"],
        target_slot=state["target_slot"],
        waiting_active=state["waiting_active"],
    )
    cv2.imshow(win_name, canvas)
    set_window_topmost(win_name)

    print("操作说明：")
    print("  F9       ：截取当前游戏部署区")
    print("  F10      ：有候选框时添加手动候选，无候选框时修正当前 target_slot")
    print("  a        ：自动检测 level logo 并添加候选 quantity 框")
    print("  d        ：完成当前 active，按候选框从右到左分配 slot 索引")
    print("  c        ：清除当前 active 的候选框")
    print("  x        ：删除离鼠标最近的候选框")
    print("  y        ：用当前鼠标位置重新固定当前 active 的 y 坐标")
    print("  n        ：跳过当前 target_slot")
    print("  z        ：撤销上一次记录")
    print("  r        ：重置当前 total 的标定（1.5 秒内按两次确认）")
    print("  s        ：保存并退出")
    print("  Esc      ：退出不保存")

    key_prev = {k: False for k in ("f9", "f10", "a", "d", "c", "x", "y", "n", "z", "r", "s", "esc")}

    while True:
        # F9：截取当前 active slot 的部署区
        if keyboard.is_pressed("f9") and not key_prev["f9"]:
            if state["active_slot"] >= args.total:
                print("所有 active_slot 已标定，可按 s 保存")
            else:
                try:
                    bar_img, window_width, window_height = capture_bar(cap)
                    state["display_bgr"] = (
                        cv2.cvtColor(bar_img, cv2.COLOR_BGRA2BGR)
                        if bar_img.shape[2] == 4
                        else bar_img.copy()
                    )
                    state["window_width"] = window_width
                    state["window_height"] = window_height
                    # 窗口尺寸变化时重新计算基准尺寸
                    base_w, base_h = base_quantity_size(window_width, window_height)
                    state["half_w"] = base_w / 2
                    state["half_h"] = base_h / 2
                    half_w_ratio = base_w / 2 / window_width
                    half_h_ratio = base_h / 2 / window_height
                    state["waiting_active"] = False
                    state["target_slot"] = _next_unassigned_target(
                        0, args.total, state["active_slot"], state["centers"]
                    )
                    state["redraw"] = True
                    print(
                        f"已截取 active={state['active_slot']}，请依次标定 target_slot，"
                        f"当前 target={state['target_slot']}"
                    )
                except Exception as e:
                    print(f"截图失败: {e}")
            key_prev["f9"] = True
        elif not keyboard.is_pressed("f9"):
            key_prev["f9"] = False

        # F10：记录鼠标位置（当前 active 下首次记录固定 y，或在自动检测模式下确定 logo 偏移）
        if keyboard.is_pressed("f10") and not key_prev["f10"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    x_ratio, y_ratio = get_mouse_window_ratio(cap)
                    if y_ratio < _BAR_CAPTURE_TOP_RATIO:
                        print("鼠标不在部署栏区域，请移到数量角标上再按 F10")
                    else:
                        a_slot = state["active_slot"]
                        if state["candidate_boxes"].get(a_slot):
                            # 当前还有候选框未最终分配，F10 用于添加手动候选
                            state["candidate_boxes"][a_slot].append((x_ratio, y_ratio))
                            state["history"].append((a_slot, -1))
                            print(
                                f"添加手动候选框 active={a_slot}，"
                                f"当前候选数 {len(state['candidate_boxes'][a_slot])}"
                            )
                            state["redraw"] = True
                        else:
                            # 候选已清空，F10 用于修正具体 slot
                            t_slot = state["target_slot"]
                            if t_slot >= args.total:
                                print("当前 active 的所有 target 已标定")
                            else:
                                if t_slot != a_slot and a_slot not in state["fixed_y_by_active"]:
                                    state["fixed_y_by_active"][a_slot] = y_ratio
                                    # 统一非 active 自身的 target 到这个 y；active 自身保留实际 y
                                    for t in list(state["centers"].get(a_slot, {}).keys()):
                                        if t != a_slot:
                                            state["centers"][a_slot][t] = (
                                                state["centers"][a_slot][t][0],
                                                state["fixed_y_by_active"][a_slot],
                                            )
                                if t_slot == a_slot:
                                    cy_ratio = y_ratio
                                else:
                                    cy_ratio = state["fixed_y_by_active"].get(a_slot, y_ratio)
                                state["centers"].setdefault(a_slot, {})[t_slot] = (x_ratio, cy_ratio)
                                state["history"].append((a_slot, t_slot))
                                fixed_note = " (y 已固定)" if a_slot in state["fixed_y_by_active"] else ""
                                print(
                                    f"标定 active={a_slot} target={t_slot} "
                                    f"center=({x_ratio:.4f}, {cy_ratio:.4f}){fixed_note}"
                                )
                                state["target_slot"] = _next_unassigned_target(
                                    t_slot + 1, args.total, a_slot, state["centers"]
                                )
                                if state["target_slot"] >= args.total:
                                    # 当前 active 完成，进入下一个 active
                                    state["active_slot"] = _next_unassigned_active(
                                        a_slot + 1, args.total, state["centers"]
                                    )
                                    state["target_slot"] = _next_unassigned_target(
                                        0, args.total, state["active_slot"], state["centers"]
                                    )
                                    state["waiting_active"] = True
                                    if state["active_slot"] < args.total:
                                        print(
                                            f"active={a_slot} 完成，请进游戏点击 "
                                            f"slot[{state['active_slot']}]，再按 F9"
                                        )
                                    else:
                                        print("所有 slot 标定完成，可按 s 保存")
                                state["redraw"] = True
                except Exception as e:
                    print(f"获取鼠标位置失败: {e}")
            key_prev["f10"] = True
        elif not keyboard.is_pressed("f10"):
            key_prev["f10"] = False

        # a：自动检测 level logo 并添加候选 quantity 框（不立即分配 slot）
        if keyboard.is_pressed("a") and not key_prev["a"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    logos = detect_level_logos(
                        state["display_bgr"],
                        state["total"],
                        state["window_width"],
                        state["window_height"],
                    )
                    state["last_logo_positions"] = logos
                    if not logos:
                        print("未检测到 level logo")
                    else:
                        a_slot = state["active_slot"]
                        state["candidate_boxes"][a_slot] = []
                        candidates = state["candidate_boxes"][a_slot]
                        added = 0
                        for lx, ly, _, scale in logos:
                            left_x = _logo_left_ratio(lx, scale, state["window_width"])
                            qx = left_x + state["logo_offset"][0]
                            qy = ly + state["logo_offset"][1]
                            candidates.append((qx, qy))
                            added += 1
                        print(
                            f"已添加 {added} 个候选 quantity 框到 active={a_slot}"
                        )
                        state["redraw"] = True
                except Exception as e:
                    print(f"自动检测失败: {e}")
            key_prev["a"] = True
        elif not keyboard.is_pressed("a"):
            key_prev["a"] = False

        # d：完成当前 active，把候选框按从右到左排序后分配 slot 索引
        if keyboard.is_pressed("d") and not key_prev["d"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                a_slot = state["active_slot"]
                candidates = state["candidate_boxes"].get(a_slot, [])
                if not candidates:
                    print("当前 active 没有候选框，请先按 a 检测")
                else:
                    # 去重并排序（从右到左）
                    unique: List[Tuple[float, float]] = []
                    seen: set = set()
                    for qx, qy in candidates:
                        key = (round(qx, 4), round(qy, 4))
                        if key not in seen:
                            seen.add(key)
                            unique.append((qx, qy))
                    unique.sort(key=lambda p: p[0], reverse=True)

                    state["centers"].setdefault(a_slot, {})
                    filled = 0
                    for i, (qx, qy) in enumerate(unique[: state["total"]]):
                        if i not in state["centers"][a_slot]:
                            state["history"].append((a_slot, i))
                        state["centers"][a_slot][i] = (qx, qy)
                        filled += 1

                    state["candidate_boxes"][a_slot] = []
                    print(f"active={a_slot} 已完成，填充 {filled} 个 slot")

                    missing = [
                        i
                        for i in range(state["total"])
                        if i not in state["centers"].get(a_slot, {})
                    ]
                    if missing:
                        print(f"缺失 slot 需要手动补: {missing}")
                        state["target_slot"] = _next_unassigned_target(
                            0, state["total"], a_slot, state["centers"]
                        )
                        state["redraw"] = True
                    else:
                        state["active_slot"] = _next_unassigned_active(
                            a_slot + 1, state["total"], state["centers"]
                        )
                        state["target_slot"] = _next_unassigned_target(
                            0, state["total"], state["active_slot"], state["centers"]
                        )
                        state["waiting_active"] = True
                        state["redraw"] = True
                        if state["active_slot"] < state["total"]:
                            print(
                                f"active={a_slot} 完成，请进游戏点击 "
                                f"slot[{state['active_slot']}]，再按 F9"
                            )
                        else:
                            print("所有 slot 标定完成，可按 s 保存")
            key_prev["d"] = True
        elif not keyboard.is_pressed("d"):
            key_prev["d"] = False

        # c：清除当前 active 的候选框
        if keyboard.is_pressed("c") and not key_prev["c"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                a_slot = state["active_slot"]
                if a_slot in state["candidate_boxes"]:
                    state["candidate_boxes"][a_slot] = []
                state["last_logo_positions"] = []
                state["redraw"] = True
                print(f"已清除 active={a_slot} 的候选框")
            key_prev["c"] = True
        elif not keyboard.is_pressed("c"):
            key_prev["c"] = False

        # x：删除离鼠标最近的候选框
        if keyboard.is_pressed("x") and not key_prev["x"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                a_slot = state["active_slot"]
                candidates = state["candidate_boxes"].get(a_slot, [])
                if not candidates:
                    print("当前没有候选框可删除")
                else:
                    try:
                        x_ratio, y_ratio = get_mouse_window_ratio(cap)
                        nearest_idx = min(
                            range(len(candidates)),
                            key=lambda i: (
                                (candidates[i][0] - x_ratio) ** 2
                                + (candidates[i][1] - y_ratio) ** 2
                            ),
                        )
                        removed = candidates.pop(nearest_idx)
                        state["history"].append((a_slot, -1))
                        state["redraw"] = True
                        print(
                            f"已删除 active={a_slot} 的候选框 "
                            f"({removed[0]:.4f}, {removed[1]:.4f})，"
                            f"剩余 {len(candidates)} 个"
                        )
                    except Exception as e:
                        print(f"删除候选框失败: {e}")
            key_prev["x"] = True
        elif not keyboard.is_pressed("x"):
            key_prev["x"] = False

        # y：重新固定当前 active 的 y 坐标
        if keyboard.is_pressed("y") and not key_prev["y"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    x_ratio, y_ratio = get_mouse_window_ratio(cap)
                    if y_ratio < _BAR_CAPTURE_TOP_RATIO:
                        print("鼠标不在部署栏区域，请移到数量角标上再按 y")
                    else:
                        a_slot = state["active_slot"]
                        state["fixed_y_by_active"][a_slot] = y_ratio
                        # 只统一非 active 自身的 target；active 自身保留实际 y
                        for t in list(state["centers"].get(a_slot, {}).keys()):
                            if t != a_slot:
                                state["centers"][a_slot][t] = (
                                    state["centers"][a_slot][t][0],
                                    state["fixed_y_by_active"][a_slot],
                                )
                        state["redraw"] = True
                        print(f"active={a_slot} 已重新固定 y 坐标: {y_ratio:.4f}")
                except Exception as e:
                    print(f"获取鼠标位置失败: {e}")
            key_prev["y"] = True
        elif not keyboard.is_pressed("y"):
            key_prev["y"] = False

        # n：跳过当前 target_slot
        if keyboard.is_pressed("n") and not key_prev["n"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                a_slot = state["active_slot"]
                t_slot = state["target_slot"]
                if t_slot < args.total:
                    print(f"跳过 active={a_slot} target={t_slot}")
                    state["target_slot"] = _next_unassigned_target(
                        t_slot + 1, args.total, a_slot, state["centers"]
                    )
                    if state["target_slot"] >= args.total:
                        state["active_slot"] = _next_unassigned_active(
                            a_slot + 1, args.total, state["centers"]
                        )
                        state["target_slot"] = _next_unassigned_target(
                            0, args.total, state["active_slot"], state["centers"]
                        )
                        state["waiting_active"] = True
                    state["redraw"] = True
            key_prev["n"] = True
        elif not keyboard.is_pressed("n"):
            key_prev["n"] = False

        # z：撤销
        if keyboard.is_pressed("z") and not key_prev["z"]:
            if state["history"]:
                last_a, last_t = state["history"].pop()
                if last_t == -1:
                    # 撤销手动添加的候选框
                    candidates = state["candidate_boxes"].get(last_a, [])
                    if candidates:
                        candidates.pop()
                        print(f"撤销 active={last_a} 的最后一个候选框")
                    state["active_slot"] = last_a
                    state["redraw"] = True
                else:
                    state["centers"].get(last_a, {}).pop(last_t, None)
                    state["active_slot"] = last_a
                    state["target_slot"] = last_t
                    state["waiting_active"] = False
                    state["redraw"] = True
                    print(f"撤销 active={last_a} target={last_t}，请重新记录")
            else:
                print("没有可撤销的标定")
            key_prev["z"] = True
        elif not keyboard.is_pressed("z"):
            key_prev["z"] = False

        # r：重置（按两次确认，避免误操作清空全部标定）
        if keyboard.is_pressed("r") and not key_prev["r"]:
            now = time.time()
            if now < state["reset_confirm_until"]:
                state["centers"].clear()
                state["history"].clear()
                state["fixed_y_by_active"].clear()
                state["logo_offset"] = (
                    _LOGO_TO_QUANTITY_DX_BASE / _BASE_WIDTH,
                    _LOGO_TO_QUANTITY_DY_BASE / _BASE_HEIGHT,
                )
                state["last_logo_positions"] = []
                state["candidate_boxes"] = {}
                state["active_slot"] = 0
                state["target_slot"] = 0
                state["waiting_active"] = True
                state["reset_confirm_until"] = 0.0
                state["redraw"] = True
                print("已重置当前 total 的全部标定")
            else:
                state["reset_confirm_until"] = now + 1.5
                print("再按一次 r 将重置当前 total 的全部标定（1.5 秒内有效）")
            key_prev["r"] = True
        elif not keyboard.is_pressed("r"):
            key_prev["r"] = False

        # s：保存
        if keyboard.is_pressed("s") and not key_prev["s"]:
            total_rois = sum(len(v) for v in state["centers"].values())
            expected = state["total"] * state["total"]
            if total_rois != expected:
                print(f"警告：当前只标定了 {total_rois}/{expected} 个位置")
            base_w = state["half_w"] * 2
            base_h = state["half_h"] * 2
            save_config(
                args.output,
                config,
                state["total"],
                state["centers"],
                base_w / 2 / state["window_width"],
                base_h / 2 / state["window_height"],
            )
            key_prev["s"] = True
            break
        elif not keyboard.is_pressed("s"):
            key_prev["s"] = False

        # Esc：退出
        if keyboard.is_pressed("esc") and not key_prev["esc"]:
            print("退出，未保存")
            key_prev["esc"] = True
            break
        elif not keyboard.is_pressed("esc"):
            key_prev["esc"] = False

        if state["redraw"]:
            canvas = draw_state(
                state["display_bgr"],
                state["centers"],
                state["half_w"],
                state["half_h"],
                state["window_width"],
                state["window_height"],
                state["total"],
                active_slot=state["active_slot"],
                target_slot=state["target_slot"],
                waiting_active=state["waiting_active"],
            )
            # 绘制最近检测到的 level logo 位置（用矩形框框出），方便确认自动检测结果
            for lx, ly, score, scale in state.get("last_logo_positions", []):
                cx, cy = from_window_ratios(lx, ly, state["window_width"], state["window_height"])
                half_w = (53 * scale) / 2
                half_h = (46 * scale) / 2
                x1 = int(round(cx - half_w))
                y1 = int(round(cy - half_h))
                x2 = int(round(cx + half_w))
                y2 = int(round(cy + half_h))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.circle(canvas, (cx, cy), 3, (0, 255, 255), -1)
                cv2.putText(
                    canvas,
                    f"{score:.2f}",
                    (x1, max(y1 - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            # 绘制候选 quantity 框（橙色），未最终分配 slot
            for qx, qy in state.get("candidate_boxes", {}).get(state["active_slot"], []):
                cx, cy = from_window_ratios(qx, qy, state["window_width"], state["window_height"])
                x1 = int(round(cx - state["half_w"]))
                y1 = int(round(cy - state["half_h"]))
                x2 = int(round(cx + state["half_w"]))
                y2 = int(round(cy + state["half_h"]))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 128, 255), 2)
                cv2.circle(canvas, (cx, cy), 3, (0, 128, 255), -1)

            cv2.imshow(win_name, canvas)
            state["redraw"] = False

        cv2.waitKey(10)
        time.sleep(0.02)

    cv2.destroyAllWindows()


def run_image_mode(
    bar_img: np.ndarray,
    args,
    config: dict,
) -> None:
    """用已有截图一次性标定所有 slot（只支持 neutral/active=0 的情况）。"""
    if bar_img.shape[2] == 4:
        display_bgr = cv2.cvtColor(bar_img, cv2.COLOR_BGRA2BGR)
    else:
        display_bgr = bar_img.copy()

    window_width, window_height = recover_window_size(bar_img)
    print(f"推断窗口尺寸: {window_width}x{window_height}")
    print(f"截图尺寸: {bar_img.shape[1]}x{bar_img.shape[0]}")

    base_w, base_h = base_quantity_size(window_width, window_height)
    half_w, half_h = base_w / 2, base_h / 2
    half_w_ratio = half_w / window_width
    half_h_ratio = half_h / window_height
    print(f"基准 ROI 尺寸: {base_w:.1f}x{base_h:.1f} (窗口比例 {half_w_ratio*2:.4f}x{half_h_ratio*2:.4f})")

    # 图片模式只标定 active=0 这一行
    centers: Dict[int, Tuple[float, float]] = {}
    history: List[int] = []
    existing = config.get("calibrations", {}).get(str(args.total), {})
    if isinstance(existing, dict):
        existing = existing.get("0", [])
    for roi in existing:
        idx = roi.get("slot")
        if idx is not None and 0 <= idx < args.total:
            centers[idx] = (roi["cx_ratio"], roi["cy_ratio"])

    state = {
        "centers": centers,
        "history": history,
        "redraw": True,
        "window_width": window_width,
        "window_height": window_height,
        "total": args.total,
        "half_w": half_w,
        "half_h": half_h,
        "display_bgr": display_bgr,
    }

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x_ratio, y_ratio = to_window_ratios(
            (x, y), param["window_width"], param["window_height"]
        )
        idx = nearest_slot_index(x_ratio, param["total"])
        param["centers"][idx] = (x_ratio, y_ratio)
        param["history"].append(idx)
        param["redraw"] = True
        print(f"标定 slot[{idx}] center=({x_ratio:.4f}, {y_ratio:.4f})")

    win_name = f"calibrate_quantity_roi (total={args.total})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse, state)

    # 先显示一次并置顶
    canvas = draw_state(
        state["display_bgr"],
        {0: state["centers"]},
        state["half_w"],
        state["half_h"],
        state["window_width"],
        state["window_height"],
        state["total"],
    )
    cv2.imshow(win_name, canvas)
    set_window_topmost(win_name)

    print("操作说明：左键点击标定，z 撤销，r 重置，s 保存退出，q/Esc 退出不保存")
    print("注意：图片模式只标定 active_slot=0 这一行，slot^2 标定请用 --live --step")

    while True:
        if state["redraw"]:
            canvas = draw_state(
                state["display_bgr"],
                {0: state["centers"]},
                state["half_w"],
                state["half_h"],
                state["window_width"],
                state["window_height"],
                state["total"],
            )
            cv2.imshow(win_name, canvas)
            state["redraw"] = False

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:
            print("退出，未保存")
            break
        elif key == ord("s"):
            if len(state["centers"]) != state["total"]:
                print(f"警告：当前只标定了 {len(state['centers'])}/{state['total']} 个 slot")
            # 保存为 active=0 的 slot^2 格式
            nested_centers = {0: state["centers"]}
            save_config(
                args.output,
                config,
                state["total"],
                nested_centers,
                half_w_ratio,
                half_h_ratio,
            )
            break
        elif key == ord("z"):
            if state["history"]:
                last = state["history"].pop()
                state["centers"].pop(last, None)
                state["redraw"] = True
                print(f"撤销 slot[{last}]")
            else:
                print("没有可撤销的标定")
        elif key == ord("r"):
            state["centers"].clear()
            state["history"].clear()
            state["redraw"] = True
            print("已重置当前 total 的标定")

    cv2.destroyAllWindows()


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """兼容中文路径的 cv2.imread 包装。

    当路径包含非 ASCII 字符时直接走 Python 二进制读取 + cv2.imdecode，
    避免 OpenCV 打印 [WARN] can't open/read file。
    """
    path_str = str(path)
    # 中文/非 ASCII 路径直接二进制读取，不走 cv2.imread
    if not all(ord(c) < 128 for c in path_str):
        try:
            with path.open("rb") as f:
                file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            return cv2.imdecode(file_bytes, flags)
        except Exception:
            return None

    img = cv2.imread(path_str, flags)
    if img is not None:
        return img
    try:
        with path.open("rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, flags)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="标定 13+ slot 时数量角标的 ROI 中心")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", "-i", type=Path, help="DEPLOY_BAR 整栏截图路径")
    source.add_argument("--live", "-l", action="store_true", help="直接截取当前游戏窗口部署区")
    parser.add_argument(
        "--step",
        action="store_true",
        help="分步热键标定模式：配合 --live，按 F9 截图、F10 记录鼠标位置",
    )
    parser.add_argument("--window-title", type=str, default="明日方舟", help="游戏窗口标题，默认 明日方舟")
    parser.add_argument("--total", "-t", type=int, required=True, help="当前截图中的总 slot 数（>=13）")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出配置文件路径，默认 data/quantity_roi_config_total{N}.json",
    )
    args = parser.parse_args()

    if args.step and not args.live:
        parser.error("--step 必须配合 --live 使用")

    if args.total <= 12:
        print("total <= 12 时使用解析器内置动态 ROI 即可，无需标定")
        raise SystemExit(0)

    if args.output is None:
        args.output = Path("data") / f"quantity_roi_config_total{args.total}.json"

    config = load_config(args.output)

    if args.image:
        if not args.image.exists():
            print(f"图片不存在: {args.image}")
            raise SystemExit(1)
        bar_img = _imread_unicode(args.image, cv2.IMREAD_UNCHANGED)
        if bar_img is None:
            print(f"读取图片失败: {args.image}")
            raise SystemExit(1)
        print(f"从文件加载: {args.image}")
        run_image_mode(bar_img, args, config)
    else:
        cap = WindowCapture(window_title=args.window_title)
        if args.step:
            run_step_mode(cap, args, config)
        else:
            bar_img, _, _ = capture_bar(cap)
            print(f"已截取游戏窗口部署区")
            run_image_mode(bar_img, args, config)


if __name__ == "__main__":
    main()
