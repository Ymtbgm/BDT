"""标定 13+ slot 时 operator_cost logo 的 X 位置，用于修正头像/费用 ROI。

核心观察：点击不同部署位时整个部署栏会发生形变，operator_cost logo 位置不仅和
目标 slot 有关，还和当前被点中的 slot（active_slot）有关。理论上需要标定
slot^2 个位置。

按键与功能与 tools/calibrate_quantity_roi.py 保持一致：
    F9  ：截取当前 active slot 的部署区
    F10 ：有候选框时添加手动候选；无候选框时记录当前 target_slot 的 logo 中心
    a   ：自动检测 operator_cost logo 并添加候选框
    d   ：完成当前 active，按候选框从右到左分配到 slot 索引（按最近 expected center）
    c   ：清除当前 active 的候选框
    x   ：删除离鼠标最近的候选框或自动检测标记
    y   ：用当前鼠标位置重新固定当前 active 的 y 坐标
    n   ：跳过当前 target_slot
    z   ：撤销上一次记录
    r   ：重置当前 active 的标定（1.5 秒内按两次确认）
    s   ：保存并退出
    Esc ：退出不保存

使用方式：
    python tools/calibrate_operator_cost_roi.py --live --step -t 13
    python tools/calibrate_operator_cost_roi.py -i deploy_bar_xxxx.png -t 13

输出：
    data/operator_cost_roi_config_total13.json
    data/operator_cost_roi_config_total14.json
    data/operator_cost_roi_config_total15.json
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

# 标定工具需要比解析器更大的顶部边距，以便在 active_slot 选中、logo 上移时仍能框选。
# core/resolver.py 中仍为 1370/1600 + 230/1600；标定输出使用窗口比例，不影响解析器使用。
_BAR_CAPTURE_TOP_RATIO = 1340 / 1600
_BAR_CAPTURE_HEIGHT_RATIO = 260 / 1600

# 费用 ROI 默认窗口比例（Y/H 不变；宽度固定 53px）
_COST_Y_RATIO = 1390 / 1600
_COST_H_RATIO = 36 / 1600
_COST_CENTER_Y_RATIO = _COST_Y_RATIO + _COST_H_RATIO / 2
_COST_W_PX = 53

# 头像中心相对 logo 中心的水平偏移（像素；头像在 logo 左侧）
_AVATAR_LOGO_DX = -40
# 费用 ROI 中心相对 logo 中心的水平偏移（像素）
_COST_LOGO_DX = -10
# 头像 ROI 可视化参数（与 core/resolver.py 一致）
_AVATAR_CENTER_Y_RATIO = 1490 / 1600
_AVATAR_SIZE_RATIO = 120 / 1600

# 模板匹配阈值（默认 0.8；若误检多则继续调高）
_TEMPLATE_MATCH_THRESHOLD = 0.78

OPERATOR_COST_LOGO_PATH = ROOT / "core" / "resource" / "operator_cost.png"

# logo 预处理：只保留纯白 (>240) 和纯黑 (<10)，中间色调置为中性灰，
# 避免 UI 背景/渐变干扰模板匹配。
_LOGO_WHITE_THRESHOLD = 240
_LOGO_BLACK_THRESHOLD = 10
_LOGO_NEUTRAL_VALUE = 128

# logo 检测搜索区域：窗口 y=1340 到 y=1390，覆盖 active_slot 上移后的 logo。
_LOGO_SEARCH_TOP_RATIO = 1340 / 1600
_LOGO_SEARCH_HEIGHT_RATIO = 50 / 1600


def _imread_unicode(path: Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    path_str = str(path)
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


def _preprocess_logo_image(img: np.ndarray) -> np.ndarray:
    """生成 4 通道 BGRA logo 掩码：有效白/黑像素不透明，中间色调透明。

    输出格式：
        - 纯白 (>240): (255, 255, 255, 255)
        - 纯黑 (<10):  (0,   0,   0,   255)
        - 中间色调:    (128, 128, 128, 0)
    Alpha 通道作为匹配掩码，中间色调不参与 NCC 计算。
    """
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    h, w = gray.shape
    out = np.zeros((h, w, 4), dtype=np.uint8)
    # 默认中间色调：中性灰 + 透明
    out[:, :, :3] = _LOGO_NEUTRAL_VALUE

    white_mask = gray > _LOGO_WHITE_THRESHOLD
    black_mask = gray < _LOGO_BLACK_THRESHOLD
    valid_mask = white_mask | black_mask

    out[white_mask] = [255, 255, 255, 255]
    out[black_mask] = [0, 0, 0, 255]
    out[valid_mask, 3] = 255
    return out


def _masked_ncc(
    search: np.ndarray,
    tmpl: np.ndarray,
    tmpl_mask: np.ndarray,
) -> np.ndarray:
    """基于模板掩码的归一化互相关。

    search: float32 灰度搜索图（无效像素已填充为全局均值）
    tmpl:   float32 灰度模板（无效像素已填充为模板均值）
    tmpl_mask: float32 模板有效像素掩码（0/1）
    返回与 cv2.matchTemplate 相同尺寸的 float32 score 图。
    """
    n = float(tmpl_mask.sum())
    if n < 1.0:
        sh, sw = search.shape
        th, tw = tmpl.shape
        return np.zeros((sh - th + 1, sw - tw + 1), dtype=np.float32)

    tmpl_mean = float((tmpl * tmpl_mask).sum()) / n
    tmpl_centered = (tmpl - tmpl_mean) * tmpl_mask
    tmpl_var = float((tmpl_centered ** 2).sum())
    if tmpl_var < 1e-6:
        sh, sw = search.shape
        th, tw = tmpl.shape
        return np.zeros((sh - th + 1, sw - tw + 1), dtype=np.float32)

    # 用 filter2D 在搜索图上滑动计算各统计量
    sum_i = cv2.filter2D(search, -1, tmpl_mask, borderType=cv2.BORDER_CONSTANT)
    sum_i2 = cv2.filter2D(search ** 2, -1, tmpl_mask, borderType=cv2.BORDER_CONSTANT)
    sum_ti = cv2.filter2D(search, -1, tmpl_centered, borderType=cv2.BORDER_CONSTANT)

    mean_i = sum_i / n
    var_i = np.maximum(sum_i2 - (sum_i ** 2) / n, 0.0)
    # tmpl_centered 已经是 (T - mean_T) * mask，因此 sum_ti 就是协方差
    cov = sum_ti

    denom = np.sqrt(tmpl_var * var_i)
    score = np.zeros_like(cov)
    valid = denom > 1e-6
    score[valid] = cov[valid] / denom[valid]
    return np.clip(score, -1.0, 1.0).astype(np.float32)


def recover_window_size(bar_img: np.ndarray) -> Tuple[int, int]:
    h_bar, w_bar = bar_img.shape[:2]
    window_width = w_bar
    window_height = int(round(h_bar / _BAR_CAPTURE_HEIGHT_RATIO))
    return window_width, window_height


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


def _draw_derived_rois(
    out: np.ndarray,
    cx_ratio: float,
    window_width: int,
    window_height: int,
    y_shift_px: int = 0,
) -> None:
    """根据 logo 中心画出费用 ROI（红）和头像 ROI（绿）。

    y_shift_px: 用于 active_slot 自身标定，整体 ROI 额外上移/下移的像素数。
    """
    h_bar, w_bar = out.shape[:2]
    y_shift_ratio = y_shift_px / window_height

    # 费用 ROI：中心在 logo X 左侧 _COST_LOGO_DX px
    cost_cx_ratio = cx_ratio + _COST_LOGO_DX / window_width
    cost_cx, cost_cy = from_window_ratios(
        cost_cx_ratio, _COST_CENTER_Y_RATIO + y_shift_ratio, window_width, window_height
    )
    cost_w = _COST_W_PX
    cost_h = int(round(_COST_H_RATIO * window_height))
    cost_x1 = max(0, min(w_bar - 1, cost_cx - cost_w // 2))
    cost_y1 = max(0, min(h_bar - 1, cost_cy - cost_h // 2))
    cost_x2 = max(cost_x1 + 1, min(w_bar, cost_cx + cost_w // 2))
    cost_y2 = max(cost_y1 + 1, min(h_bar, cost_cy + cost_h // 2))
    cv2.rectangle(out, (cost_x1, cost_y1), (cost_x2, cost_y2), (0, 0, 255), 2)
    cv2.circle(out, (cost_cx, cost_cy), 3, (0, 0, 255), -1)

    # 头像 ROI：中心在 logo X 左侧 _AVATAR_LOGO_DX px
    avatar_cx_ratio = cx_ratio + _AVATAR_LOGO_DX / window_width
    avatar_cx, avatar_cy = from_window_ratios(
        avatar_cx_ratio, _AVATAR_CENTER_Y_RATIO + y_shift_ratio, window_width, window_height
    )
    avatar_size = int(round(_AVATAR_SIZE_RATIO * window_height))
    avatar_x1 = max(0, min(w_bar - 1, avatar_cx - avatar_size // 2))
    avatar_y1 = max(0, min(h_bar - 1, avatar_cy - avatar_size // 2))
    avatar_x2 = max(avatar_x1 + 1, min(w_bar, avatar_cx + avatar_size // 2))
    avatar_y2 = max(avatar_y1 + 1, min(h_bar, avatar_cy + avatar_size // 2))
    cv2.rectangle(out, (avatar_x1, avatar_y1), (avatar_x2, avatar_y2), (0, 255, 0), 2)
    cv2.circle(out, (avatar_cx, avatar_cy), 3, (0, 255, 0), -1)


def dynamic_slot_center_x_ratio(slot_index: int, total_slots: int) -> float:
    return 1.0 - (slot_index + 0.5) / total_slots


def nearest_slot_index(x_ratio: float, total_slots: int) -> int:
    return min(
        range(total_slots),
        key=lambda i: abs(x_ratio - dynamic_slot_center_x_ratio(i, total_slots)),
    )


def detect_operator_cost_logos(
    bar_img: np.ndarray,
    window_width: int,
    window_height: int,
    threshold: float = _TEMPLATE_MATCH_THRESHOLD,
) -> List[Tuple[float, float, float, float]]:
    """检测 bar_img 中的 operator_cost logo，返回窗口比例 (cx_ratio, cy_ratio, score, scale)。

    结果按 x 从右到左排序（slot 0 在最前）。
    """
    if not OPERATOR_COST_LOGO_PATH.exists():
        raise FileNotFoundError(f"未找到 operator_cost logo 模板: {OPERATOR_COST_LOGO_PATH}")

    tmpl = _imread_unicode(OPERATOR_COST_LOGO_PATH, cv2.IMREAD_UNCHANGED)
    if tmpl is None:
        raise ValueError(f"无法读取 operator_cost logo 模板: {OPERATOR_COST_LOGO_PATH}")
    tmpl_bgra = _preprocess_logo_image(tmpl)
    tmpl_gray = cv2.cvtColor(tmpl_bgra, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    tmpl_mask = (tmpl_bgra[:, :, 3] > 0).astype(np.float32)
    tmpl_mean = float(np.mean(tmpl_gray[tmpl_mask > 0])) if np.any(tmpl_mask > 0) else _LOGO_NEUTRAL_VALUE
    tmpl_gray[tmpl_mask == 0] = tmpl_mean

    # 只在 y=1365~1400 的窄带内搜索 logo，减少背景干扰
    h_bar, w_bar = bar_img.shape[:2]
    search_top_bar = int(round(
        h_bar * (_LOGO_SEARCH_TOP_RATIO - _BAR_CAPTURE_TOP_RATIO) / _BAR_CAPTURE_HEIGHT_RATIO
    ))
    search_bottom_bar = int(round(
        h_bar * (_LOGO_SEARCH_TOP_RATIO + _LOGO_SEARCH_HEIGHT_RATIO - _BAR_CAPTURE_TOP_RATIO) / _BAR_CAPTURE_HEIGHT_RATIO
    ))
    search_top_bar = max(0, search_top_bar)
    search_bottom_bar = min(h_bar, search_bottom_bar)
    search_img = bar_img[search_top_bar:search_bottom_bar, :]

    if len(search_img.shape) == 3 and search_img.shape[2] in (3, 4):
        search_gray = cv2.cvtColor(
            search_img,
            cv2.COLOR_BGRA2GRAY if search_img.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
        )
    else:
        search_gray = search_img
    search_bgra = _preprocess_logo_image(search_gray)
    search_gray = cv2.cvtColor(search_bgra, cv2.COLOR_BGRA2GRAY).astype(np.float32)
    search_mask = (search_bgra[:, :, 3] > 0).astype(np.float32)
    search_mean = float(np.mean(search_gray[search_mask > 0])) if np.any(search_mask > 0) else _LOGO_NEUTRAL_VALUE
    search_gray[search_mask == 0] = search_mean

    scale_ref = window_height / 1600
    scales = [scale_ref * s for s in np.linspace(0.8, 1.2, 9)]

    tmpl_h, tmpl_w = tmpl_gray.shape
    detections: List[Tuple[float, float, float, float]] = []
    for scale in scales:
        resized_tmpl = cv2.resize(tmpl_gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        resized_mask = cv2.resize(tmpl_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        resized_mask = np.clip(resized_mask, 0.0, 1.0)
        rh, rw = resized_tmpl.shape
        if rh > search_gray.shape[0] or rw > search_gray.shape[1]:
            continue
        result = _masked_ncc(search_gray, resized_tmpl, resized_mask)
        loc = np.where(result >= threshold)
        for pt in zip(*loc[::-1]):
            cx = pt[0] + rw / 2
            cy = pt[1] + rh / 2
            score = result[pt[1], pt[0]]
            detections.append((cx, cy, score, scale))

    if not detections:
        return []

    detections.sort(key=lambda x: x[2], reverse=True)
    base_size = max(tmpl_w, tmpl_h) * scale_ref
    min_dist = base_size * 0.7
    filtered: List[Tuple[float, float, float, float]] = []
    for cx, cy, score, scale in detections:
        too_close = False
        for fx, fy, _, _ in filtered:
            if np.hypot(cx - fx, cy - fy) < min_dist:
                too_close = True
                break
        if not too_close:
            filtered.append((cx, cy, score, scale))

    filtered.sort(key=lambda p: p[0], reverse=True)
    bar_top_px = window_height * _BAR_CAPTURE_TOP_RATIO
    return [
        (cx / window_width, (cy + search_top_bar + bar_top_px) / window_height, score, scale)
        for cx, cy, score, scale in filtered
    ]


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
    candidate_boxes: Optional[Dict[int, List[Tuple[float, float, bool]]]] = None,
    last_logo_positions: Optional[List[Tuple[float, float, float, float]]] = None,
) -> np.ndarray:
    out = canvas.copy()
    h_bar, w_bar = out.shape[:2]
    candidate_boxes = candidate_boxes or {}
    last_logo_positions = last_logo_positions or []

    # 动态 slot 中心线
    for i in range(total_slots):
        cx_ratio = dynamic_slot_center_x_ratio(i, total_slots)
        x = int(round(cx_ratio * window_width))
        if not (0 <= x < w_bar):
            continue
        if i == active_slot:
            color = (0, 0, 255)
            thickness = 2
        elif i == target_slot and not waiting_active:
            color = (255, 0, 0)
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

    # 已标定的 logo 位置：只显示当前 active_slot，避免其他 active 的框过多干扰
    for a_slot, targets in sorted(centers.items()):
        if a_slot != active_slot:
            continue
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
            is_self = a_slot == t_slot
            color = (0, 255, 0)
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
            if is_self:
                # active_slot 自身标定：画十字标记，便于识别
                cross_len = 8
                cv2.line(out, (cx - cross_len, cy), (cx + cross_len, cy), color, 2)
                cv2.line(out, (cx, cy - cross_len), (cx, cy + cross_len), color, 2)
            _draw_derived_rois(
                out,
                cx_ratio,
                window_width,
                window_height,
                y_shift_px=-25 if is_self else 0,
            )

    # 最近检测到的 logo 位置（黄色）
    for lx, ly, score, scale in last_logo_positions:
        cx, cy = from_window_ratios(lx, ly, window_width, window_height)
        half_w_px = (23 * scale) / 2
        half_h_px = (24 * scale) / 2
        x1 = int(round(cx - half_w_px))
        y1 = int(round(cy - half_h_px))
        x2 = int(round(cx + half_w_px))
        y2 = int(round(cy + half_h_px))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), 3, (0, 255, 255), -1)
        cv2.putText(
            out,
            f"{score:.2f}",
            (x1, max(y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        _draw_derived_rois(out, lx, window_width, window_height)

    # 候选框（橙色，未最终分配）
    for cand in candidate_boxes.get(active_slot, []):
        qx, qy = cand[:2]
        is_self = bool(cand[2]) if len(cand) >= 3 else False
        cx, cy = from_window_ratios(qx, qy, window_width, window_height)
        x1 = int(round(cx - half_w))
        y1 = int(round(cy - half_h))
        x2 = int(round(cx + half_w))
        y2 = int(round(cy + half_h))
        x1 = max(0, min(w_bar - 1, x1))
        y1 = max(0, min(h_bar - 1, y1))
        x2 = max(x1 + 1, min(w_bar, x2))
        y2 = max(y1 + 1, min(h_bar, y2))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 128, 255), 2)
        cv2.circle(out, (cx, cy), 3, (0, 128, 255), -1)
        _draw_derived_rois(
            out, qx, window_width, window_height,
            y_shift_px=-25 if is_self else 0,
        )

    # 操作提示
    if active_slot >= 0:
        if waiting_active:
            status = f"请进游戏点击 slot[{active_slot}] 进入部署状态，然后按 F9 截图"
        else:
            status = f"active={active_slot} target={target_slot}，a检测 d分配 c清除 x删除 y固定y n跳过"
        hints = [
            "F9: 截图  |  F10: 手动候选/记录  |  a: 检测  |  d: 分配  |  c: 清除  |  x: 删除  |  y: 固定y  |  n: 跳过  |  o: active上移25px  |  z: 撤销  |  r: 重置  |  s: 保存  |  Esc: 退出",
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
    return {"version": "1.0", "base_size": {}, "calibrations": {}}


def save_config(
    path: Path,
    config: dict,
    total_slots: int,
    centers: Dict[int, Dict[int, Tuple[float, float]]],
    half_w_ratio: float,
    half_h_ratio: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
    return full_img[y : y + roi_h, x : x + roi_w], window_width, window_height


def get_mouse_window_ratio(cap: WindowCapture) -> Tuple[float, float]:
    hwnd = cap._find_hwnd()
    cx_screen, cy_screen = win32gui.GetCursorPos()
    cx_client, cy_client = win32gui.ScreenToClient(hwnd, (cx_screen, cy_screen))
    window_width, window_height = cap.get_window_size()
    return cx_client / window_width, cy_client / window_height


def set_window_topmost(win_name: str) -> None:
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


def assign_candidates_to_slots(
    candidates: List[Tuple[float, float, bool]],
    total_slots: int,
) -> Dict[int, Tuple[float, float]]:
    """把候选 logo 按最终 X 位置从左到右分配到 slot_index。

    候选框格式支持 3 元组 (cx, cy, is_self)，分配时只使用 cx。
    排序后最左侧的候选分配给最左侧的 slot（即 slot_index 最大），
    最右侧的候选分配给最右侧的 slot（即 slot_index 0）。
    """
    assignments: Dict[int, Tuple[float, float]] = {}
    # 按 X 从小到大（从左到右）排序
    sorted_cands = sorted(candidates, key=lambda c: c[0])
    # 从左到右依次对应 slot total_slots-1, total_slots-2, ..., 0
    for i, cand in enumerate(sorted_cands):
        if i >= total_slots:
            break
        slot = total_slots - 1 - i
        assignments[slot] = cand[:2]
    return assignments


def run_step_mode(cap: WindowCapture, args, config: dict) -> None:
    bar_img, window_width, window_height = capture_bar(cap)
    half_w, half_h = 23 / 2, 24 / 2  # logo 原始半宽高
    half_w_ratio = half_w / 2560
    half_h_ratio = half_h / 1600

    print(f"推断窗口尺寸: {window_width}x{window_height}")
    print(f"基准 logo 尺寸: {half_w*2:.0f}x{half_h*2:.0f} (窗口比例 {half_w_ratio*2:.4f}x{half_h_ratio*2:.4f})")

    centers: Dict[int, Dict[int, Tuple[float, float]]] = {}
    history: List[Tuple[int, int]] = []
    fixed_y_by_active: Dict[int, float] = {}
    candidate_boxes: Dict[int, List[Tuple[float, float, bool]]] = {}

    existing = config.get("calibrations", {}).get(str(args.total), {})
    if isinstance(existing, list):
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
    last_logo_positions: List[Tuple[float, float, float, float]] = []

    win_name = f"calibrate_operator_cost_roi (total={args.total})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, lambda *a, **k: None)

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
        "candidate_boxes": candidate_boxes,
        "last_logo_positions": last_logo_positions,
        "reset_confirm_until": 0.0,
        "threshold": args.threshold,
    }

    def draw():
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
            candidate_boxes=state["candidate_boxes"],
            last_logo_positions=state["last_logo_positions"],
        )
        cv2.imshow(win_name, canvas)

    draw()
    set_window_topmost(win_name)

    print("操作说明：")
    print("  F9  ：截取当前 active slot 的部署区")
    print("  F10 ：有候选框时添加手动候选；无候选框时记录当前 target_slot 的 logo 中心")
    print("  a   ：自动检测 operator_cost logo 并添加候选框")
    print("  d   ：完成当前 active，按候选框从右到左分配到 slot")
    print("  c   ：清除当前 active 的候选框")
    print("  x   ：删除离鼠标最近的候选框或自动检测标记")
    print("  y   ：用当前鼠标位置重新固定当前 active 的 y 坐标")
    print("  n   ：跳过当前 target_slot")
    print("  o   ：添加 active_slot 自身候选（Y 上移 25 像素），需按 d 分配")
    print("  z   ：撤销上一次记录")
    print("  r   ：重置当前 active（1.5 秒内按两次确认）")
    print("  s   ：保存并退出")
    print("  Esc ：退出不保存")

    key_prev = {k: False for k in ("f9", "f10", "a", "d", "c", "x", "y", "n", "z", "r", "s", "esc", "o")}

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
                    state["half_w"] = (23 / 2) * (window_height / 1600)
                    state["half_h"] = (24 / 2) * (window_height / 1600)
                    state["waiting_active"] = False
                    state["target_slot"] = _next_unassigned_target(
                        0, args.total, state["active_slot"], state["centers"]
                    )
                    state["candidate_boxes"].pop(state["active_slot"], None)
                    state["last_logo_positions"] = []
                    state["redraw"] = True
                    print(
                        f"已截取 active={state['active_slot']}，"
                        f"当前 target={state['target_slot']}"
                    )
                except Exception as e:
                    print(f"截图失败: {e}")
            key_prev["f9"] = True
        elif not keyboard.is_pressed("f9"):
            key_prev["f9"] = False

        # F10：有候选框时添加手动候选；无候选框时记录当前 target_slot
        if keyboard.is_pressed("f10") and not key_prev["f10"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    x_ratio, y_ratio = get_mouse_window_ratio(cap)
                    a_slot = state["active_slot"]
                    if state["candidate_boxes"].get(a_slot):
                        state["candidate_boxes"][a_slot].append((x_ratio, y_ratio, False))
                        state["history"].append((a_slot, -1))
                        state["redraw"] = True
                        print(
                            f"添加手动候选框 active={a_slot}，"
                            f"当前候选数 {len(state['candidate_boxes'][a_slot])}"
                        )
                    else:
                        t_slot = state["target_slot"]
                        if t_slot >= args.total:
                            print("当前 active 的所有 target 已标定")
                        else:
                            if t_slot != a_slot and a_slot not in state["fixed_y_by_active"]:
                                state["fixed_y_by_active"][a_slot] = y_ratio
                                for t in list(state["centers"].get(a_slot, {}).keys()):
                                    if t != a_slot:
                                        state["centers"][a_slot][t] = (
                                            state["centers"][a_slot][t][0],
                                            state["fixed_y_by_active"][a_slot],
                                        )
                            cy_ratio = y_ratio if t_slot == a_slot else state["fixed_y_by_active"].get(a_slot, y_ratio)
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
                                state["active_slot"] = _next_unassigned_active(
                                    a_slot + 1, args.total, state["centers"]
                                )
                                state["target_slot"] = _next_unassigned_target(
                                    0, args.total, state["active_slot"], state["centers"]
                                )
                                state["waiting_active"] = True
                                state["candidate_boxes"].pop(a_slot, None)
                                state["last_logo_positions"] = []
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

        # a：自动检测 operator_cost logo 并添加候选框
        if keyboard.is_pressed("a") and not key_prev["a"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    logos = detect_operator_cost_logos(
                        state["display_bgr"],
                        state["window_width"],
                        state["window_height"],
                        threshold=state["threshold"],
                    )
                    state["last_logo_positions"] = logos
                    if not logos:
                        print("未检测到 operator_cost logo")
                    else:
                        a_slot = state["active_slot"]
                        state["candidate_boxes"][a_slot] = [
                            (lx, ly, False) for lx, ly, _, _ in logos
                        ]
                        print(
                            f"已添加 {len(logos)} 个候选 logo 框到 active={a_slot}"
                        )
                        state["redraw"] = True
                except Exception as e:
                    print(f"自动检测失败: {e}")
            key_prev["a"] = True
        elif not keyboard.is_pressed("a"):
            key_prev["a"] = False

        # d：完成当前 active，把候选框按最近 slot 分配
        if keyboard.is_pressed("d") and not key_prev["d"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                a_slot = state["active_slot"]
                candidates = state["candidate_boxes"].get(a_slot, [])
                if not candidates:
                    print("当前 active 没有候选框，请先按 a 检测或 F10 手动添加")
                else:
                    unique: List[Tuple[float, float, bool]] = []
                    seen: set = set()
                    for cand in candidates:
                        cx, cy = cand[:2]
                        key = (round(cx, 4), round(cy, 4))
                        if key not in seen:
                            seen.add(key)
                            unique.append(cand)

                    assignments = assign_candidates_to_slots(unique, state["total"])
                    state["centers"].setdefault(a_slot, {})
                    filled = 0
                    for slot, (cx, cy) in assignments.items():
                        if slot in state["centers"][a_slot]:
                            continue  # 已手动标定，跳过
                        state["history"].append((a_slot, slot))
                        # 非 active 自身 target 使用固定 y
                        if slot != a_slot and a_slot in state["fixed_y_by_active"]:
                            cy = state["fixed_y_by_active"][a_slot]
                        state["centers"][a_slot][slot] = (cx, cy)
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
                        state["candidate_boxes"].pop(a_slot, None)
                        state["last_logo_positions"] = []
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

        # x：删除离鼠标最近的候选框或检测标记
        if keyboard.is_pressed("x") and not key_prev["x"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    x_ratio, y_ratio = get_mouse_window_ratio(cap)
                    a_slot = state["active_slot"]
                    candidates = state["candidate_boxes"].get(a_slot, [])
                    logos = state["last_logo_positions"]

                    # 合并候选框和黄色检测标记，删除离鼠标最近的一个
                    deletables: List[Tuple[str, int, float, float]] = []
                    for i, cand in enumerate(candidates):
                        deletables.append(("candidate", i, cand[0], cand[1]))
                    for i, logo in enumerate(logos):
                        deletables.append(("logo", i, logo[0], logo[1]))

                    if not deletables:
                        print("当前没有可删除的框")
                    else:
                        nearest = min(
                            deletables,
                            key=lambda it: (it[2] - x_ratio) ** 2
                            + (it[3] - y_ratio) ** 2,
                        )
                        src, idx = nearest[0], nearest[1]
                        if src == "candidate":
                            removed = candidates.pop(idx)
                            state["history"].append((a_slot, -1))
                            print(
                                f"已删除 active={a_slot} 的候选框 "
                                f"({removed[0]:.4f}, {removed[1]:.4f})，"
                                f"剩余 {len(candidates)} 个"
                            )
                        else:
                            removed = logos.pop(idx)
                            print(
                                f"已删除检测标记 "
                                f"({removed[0]:.4f}, {removed[1]:.4f})，"
                                f"剩余 {len(logos)} 个"
                            )
                        state["redraw"] = True
                except Exception as e:
                    print(f"删除失败: {e}")
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
                        print("鼠标不在部署栏区域，请移到 logo 上再按 y")
                    else:
                        a_slot = state["active_slot"]
                        state["fixed_y_by_active"][a_slot] = y_ratio
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
                        state["candidate_boxes"].pop(a_slot, None)
                        state["last_logo_positions"] = []
                        if state["active_slot"] < args.total:
                            print(
                                f"active={a_slot} 完成，请进游戏点击 "
                                f"slot[{state['active_slot']}]，再按 F9"
                            )
                        else:
                            print("所有 slot 标定完成，可按 s 保存")
                    state["redraw"] = True
            key_prev["n"] = True
        elif not keyboard.is_pressed("n"):
            key_prev["n"] = False

        # o：添加 active_slot 自身候选，Y 坐标上移 25 像素（按 d 分配后生效）
        if keyboard.is_pressed("o") and not key_prev["o"]:
            if state["waiting_active"]:
                print("请先按 F9 截取部署区")
            else:
                try:
                    x_ratio, y_ratio = get_mouse_window_ratio(cap)
                    a_slot = state["active_slot"]
                    cy_ratio = y_ratio - 25 / state["window_height"]
                    state["candidate_boxes"].setdefault(a_slot, []).append(
                        (x_ratio, cy_ratio, True)
                    )
                    state["history"].append((a_slot, -1))
                    state["redraw"] = True
                    print(
                        f"添加 active={a_slot} 自身候选，Y 上移 25px: "
                        f"({x_ratio:.4f}, {cy_ratio:.4f})，"
                        f"当前候选数 {len(state['candidate_boxes'][a_slot])}"
                    )
                except Exception as e:
                    print(f"获取鼠标位置失败: {e}")
            key_prev["o"] = True
        elif not keyboard.is_pressed("o"):
            key_prev["o"] = False

        # z：撤销
        if keyboard.is_pressed("z") and not key_prev["z"]:
            if state["history"]:
                last_a, last_t = state["history"].pop()
                if last_t == -1:
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

        # r：重置当前 active
        if keyboard.is_pressed("r") and not key_prev["r"]:
            now = time.time()
            if now < state["reset_confirm_until"]:
                active = state["active_slot"]
                state["centers"].pop(active, None)
                state["history"] = [(a, t) for a, t in state["history"] if a != active]
                state["fixed_y_by_active"].pop(active, None)
                state["candidate_boxes"].pop(active, None)
                state["last_logo_positions"] = []
                state["target_slot"] = 0
                state["waiting_active"] = True
                state["reset_confirm_until"] = 0.0
                state["redraw"] = True
                print(f"已重置 active={active} 的标定")
            else:
                state["reset_confirm_until"] = now + 1.5
                print("再按一次 r 将重置当前 active 的标定（1.5 秒内有效）")
            key_prev["r"] = True
        elif not keyboard.is_pressed("r"):
            key_prev["r"] = False

        # s：保存
        if keyboard.is_pressed("s") and not key_prev["s"]:
            total_rois = sum(len(v) for v in state["centers"].values())
            expected = state["total"] * state["total"]
            if total_rois != expected:
                print(f"警告：当前只标定了 {total_rois}/{expected} 个位置")
            save_config(
                args.output,
                config,
                state["total"],
                state["centers"],
                state["half_w"] / state["window_width"],
                state["half_h"] / state["window_height"],
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
            draw()
            state["redraw"] = False

        cv2.waitKey(10)
        time.sleep(0.02)

    cv2.destroyAllWindows()


def run_image_mode(bar_img: np.ndarray, args, config: dict) -> None:
    """图片模式只标定 active_slot=0。支持 a 自动检测。"""
    if bar_img.shape[2] == 4:
        display_bgr = cv2.cvtColor(bar_img, cv2.COLOR_BGRA2BGR)
    else:
        display_bgr = bar_img.copy()

    window_width, window_height = recover_window_size(bar_img)
    print(f"推断窗口尺寸: {window_width}x{window_height}")

    half_w = (23 / 2) * (window_height / 1600)
    half_h = (24 / 2) * (window_height / 1600)

    centers: Dict[int, Tuple[float, float]] = {}
    history: List[int] = []
    existing = config.get("calibrations", {}).get(str(args.total), {})
    if isinstance(existing, dict):
        existing = existing.get("0", [])
    for roi in existing:
        idx = roi.get("slot")
        if idx is not None and 0 <= idx < args.total:
            centers[idx] = (roi["cx_ratio"], roi["cy_ratio"])

    candidate_boxes: List[Tuple[float, float]] = []
    last_logos: List[Tuple[float, float, float, float]] = []

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
        "candidate_boxes": candidate_boxes,
        "last_logos": last_logos,
        "threshold": args.threshold,
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

    win_name = f"calibrate_operator_cost_roi (total={args.total})"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse, state)
    set_window_topmost(win_name)

    def draw():
        canvas = draw_state(
            state["display_bgr"],
            {0: state["centers"]},
            state["half_w"],
            state["half_h"],
            state["window_width"],
            state["window_height"],
            state["total"],
            active_slot=0,
            target_slot=-1,
            waiting_active=False,
            candidate_boxes={0: state["candidate_boxes"]},
            last_logo_positions=state["last_logos"],
        )
        cv2.imshow(win_name, canvas)

    draw()
    print("操作说明：")
    print("  左键：手动标定 slot")
    print("  a   ：自动检测 operator_cost logo 并添加候选")
    print("  d   ：按候选分配 slot")
    print("  c   ：清除候选")
    print("  z   ：撤销")
    print("  r   ：重置")
    print("  s   ：保存退出")
    print("  q/Esc：退出不保存")
    print("注意：图片模式只标定 active_slot=0")

    while True:
        if state["redraw"]:
            draw()
            state["redraw"] = False

        key = cv2.waitKey(50) & 0xFF
        if key == ord("q") or key == 27:
            print("退出，未保存")
            break
        elif key == ord("s"):
            if len(state["centers"]) != state["total"]:
                print(f"警告：当前只标定了 {len(state['centers'])}/{state['total']} 个 slot")
            save_config(
                args.output,
                config,
                state["total"],
                {0: state["centers"]},
                state["half_w"] / state["window_width"],
                state["half_h"] / state["window_height"],
            )
            break
        elif key == ord("a"):
            try:
                logos = detect_operator_cost_logos(
                    state["display_bgr"],
                    state["window_width"],
                    state["window_height"],
                    threshold=state["threshold"],
                )
                state["last_logos"] = logos
                state["candidate_boxes"] = [(lx, ly) for lx, ly, _, _ in logos]
                print(f"检测到 {len(logos)} 个 operator_cost logo")
                state["redraw"] = True
            except Exception as e:
                print(f"自动检测失败: {e}")
        elif key == ord("d"):
            if not state["candidate_boxes"]:
                print("当前没有候选框，请先按 a 检测")
            else:
                assignments = assign_candidates_to_slots(
                    state["candidate_boxes"], state["total"]
                )
                for slot, (cx, cy) in assignments.items():
                    state["centers"][slot] = (cx, cy)
                    if slot not in state["history"]:
                        state["history"].append(slot)
                state["candidate_boxes"] = []
                state["redraw"] = True
                print(f"已分配 {len(assignments)} 个 slot")
        elif key == ord("c"):
            state["candidate_boxes"] = []
            state["last_logos"] = []
            state["redraw"] = True
            print("已清除候选框")
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
            state["candidate_boxes"] = []
            state["last_logos"] = []
            state["redraw"] = True
            print("已重置当前 total 的标定")

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="标定 13+ slot 时 operator_cost logo 的 X 位置"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image", "-i", type=Path, help="DEPLOY_BAR 整栏截图路径"
    )
    source.add_argument(
        "--live", "-l", action="store_true", help="直接截取当前游戏窗口部署区"
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="分步热键标定模式：配合 --live，按 F9 截图、F10 记录",
    )
    parser.add_argument(
        "--window-title", type=str, default="明日方舟", help="游戏窗口标题，默认 明日方舟"
    )
    parser.add_argument(
        "--total", "-t", type=int, required=True, help="当前截图中的总 slot 数（>=13）"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="输出配置文件路径，默认 data/operator_cost_roi_config_total{N}.json",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_TEMPLATE_MATCH_THRESHOLD,
        help=f"logo 自动检测阈值，默认 {_TEMPLATE_MATCH_THRESHOLD}；误检多则调高",
    )
    args = parser.parse_args()

    if args.step and not args.live:
        parser.error("--step 必须配合 --live 使用")

    if args.total <= 12:
        print("total <= 12 时使用解析器内置动态 ROI 即可，无需标定")
        raise SystemExit(0)

    if args.output is None:
        args.output = Path("data") / f"operator_cost_roi_config_total{args.total}.json"

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
            print("已截取游戏窗口部署区")
            run_image_mode(bar_img, args, config)


if __name__ == "__main__":
    main()
