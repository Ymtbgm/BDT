import os
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.capture import WindowCapture
from core.ocr_engine import OCREngine
import core.constants as constants


def recognize_operator_costs(
    capture: WindowCapture,
    ocr: OCREngine,
    operators: List[str],
    items: List[str],
    support_count: int = 0,
    direct_start: bool = False,
    debug: bool = False,
) -> Tuple[Dict[str, int], bool]:
    """按部署栏格子精确裁剪并 OCR 识别干员费用。

    返回 (costs, manual_support_detected)。
    costs: {operator_name: cost}，识别失败时可能为空字典。
    manual_support_detected: 直接开始作战且未勾选借助战时，若检测到最右侧为助战干员则返回 True。
    """
    if not operators:
        return {}, False

    w, h = capture.get_window_size()
    left = capture.monitor.get("left", 0)
    top = capture.monitor.get("top", 0)
    ratios = constants.DEPLOY_BAR_COST_ROI_RATIOS
    y = int(h * ratios[1]) + top
    rh = int(h * ratios[3])

    operators_to_recognize = operators[:-support_count] if support_count > 0 else operators
    num_operators = len(operators)
    num_items = len(items)
    total = num_operators + num_items
    cell_w = w / 12 if total <= 12 else w / total

    session_id = int(time.time() * 1000)
    session_dir = os.path.join("debug", "recorder", "operator_cost_ocr", str(session_id))
    if debug:
        os.makedirs(session_dir, exist_ok=True)
        print(
            f"[部署栏OCR] 会话={session_id} 窗口=({w}x{h}) 格子宽={cell_w:.1f} "
            f"干员数={num_operators} 道具数={num_items} 助战={support_count}"
        )

    mapping = {}
    for i, name in enumerate(operators_to_recognize):
        bar_index = total - 1 - i
        cx = w - cell_w * (bar_index + 0.5)
        x = int(cx) + left
        rw = 53
        try:
            img = capture.capture_roi(x, y, rw, rh)
        except Exception as e:
            if debug:
                print(f"[部署栏OCR] {name}: 截取 ROI 失败: {e}")
            continue

        raw_path = os.path.join(session_dir, f"{bar_index:02d}_raw.png") if debug else None
        fixed_path = os.path.join(session_dir, f"{bar_index:02d}_fixed.png") if debug else None
        inv_path = os.path.join(session_dir, f"{bar_index:02d}_inv.png") if debug else None

        fixed_img = preprocess_cost_image(img)
        inv_img = preprocess_cost_image_inv(img)

        fixed_result = extract_cost_with_conf(
            ocr.recognize(fixed_img, min_confidence=0.5), min_conf=0.5
        )
        inv_result = extract_cost_with_conf(
            ocr.recognize(inv_img, min_confidence=0.5), min_conf=0.5
        )

        if debug:
            os.makedirs(session_dir, exist_ok=True)
            cv2.imwrite(raw_path, img)
            cv2.imwrite(fixed_path, fixed_img)
            cv2.imwrite(inv_path, inv_img)
            fixed_str = f"{fixed_result[0]}({fixed_result[1]:.2f})" if fixed_result else "失败"
            inv_str = f"{inv_result[0]}({inv_result[1]:.2f})" if inv_result else "失败"

        chosen = None
        chosen_source = None
        if fixed_result:
            chosen = fixed_result[0]
            chosen_source = "固定阈值"
        elif inv_result:
            chosen = inv_result[0]
            chosen_source = "反色"

        if chosen is not None:
            mapping[name] = chosen
            if debug:
                print(f"[部署栏OCR] {name}: 固定阈值={fixed_str}, 反色={inv_str} → {chosen} ({chosen_source})")
        elif debug:
            print(f"[部署栏OCR] {name}: 固定阈值={fixed_str}, 反色={inv_str} → 失败")

    expected = len(operators_to_recognize)
    if len(mapping) < expected:
        if debug:
            print(
                f"[部署栏OCR] 仅识别到 {len(mapping)}/{expected} 个费用，"
                f"回退到初始序号排序"
            )
        return {}, False

    # 自动检测手动借用的助战干员：仅直接开始作战时启用。
    manual_support = False
    if (
        direct_start
        and support_count == 0
        and len(operators) >= 2
    ):
        rightmost_name = operators[-1]
        left_neighbor_name = operators[-2]
        if rightmost_name in mapping and left_neighbor_name in mapping:
            if mapping[rightmost_name] < mapping[left_neighbor_name]:
                manual_support = True
                if debug:
                    print(
                        f"[部署栏OCR] 检测到手动助战: {rightmost_name}"
                        f"({mapping[rightmost_name]}) < {left_neighbor_name}"
                        f"({mapping[left_neighbor_name]}), 已固定为最右"
                    )

    if debug:
        print(f"[部署栏OCR] 识别费用: {mapping}")
    return mapping, manual_support


def recognize_costs_by_bar_index(
    capture: WindowCapture,
    ocr: OCREngine,
    name_to_bar_index: Dict[str, int],
    items: List[str],
    operators: List[str],
    debug: bool = False,
) -> Dict[str, int]:
    """按已知的 name->bar_index 映射读取各格子费用。未匹配或识别失败不返回。

    Args:
        name_to_bar_index: 干员名到部署栏全局索引（0=最右侧）的映射。
        items: 道具名称列表，用于计算总格子数。
        operators: 干员名称列表，用于计算总格子数。
    """
    if not name_to_bar_index:
        return {}

    w, h = capture.get_window_size()
    left = capture.monitor.get("left", 0)
    top = capture.monitor.get("top", 0)
    ratios = constants.DEPLOY_BAR_COST_ROI_RATIOS
    y = int(h * ratios[1]) + top
    rh = int(h * ratios[3])

    total = len(operators) + len(items)
    if total == 0:
        return {}
    cell_w = w / 12 if total <= 12 else w / total

    session_id = int(time.time() * 1000)
    session_dir = os.path.join("debug", "recorder", "operator_cost_ocr", str(session_id))
    if debug:
        os.makedirs(session_dir, exist_ok=True)
        print(
            f"[部署栏OCR] 按位置读取 会话={session_id} 窗口=({w}x{h}) 格子宽={cell_w:.1f} "
            f"总格子={total} 待读={len(name_to_bar_index)}"
        )

    mapping: Dict[str, int] = {}
    for name, bar_index in name_to_bar_index.items():
        cx = w - cell_w * (bar_index + 0.5)
        x = int(cx) + left
        rw = 53
        try:
            img = capture.capture_roi(x, y, rw, rh)
        except Exception as e:
            if debug:
                print(f"[部署栏OCR] {name} (bar[{bar_index}]): 截取 ROI 失败: {e}")
            continue

        if debug:
            os.makedirs(session_dir, exist_ok=True)

        fixed_img = preprocess_cost_image(img)
        inv_img = preprocess_cost_image_inv(img)

        fixed_result = extract_cost_with_conf(
            ocr.recognize(fixed_img, min_confidence=0.5), min_conf=0.5
        )
        inv_result = extract_cost_with_conf(
            ocr.recognize(inv_img, min_confidence=0.5), min_conf=0.5
        )

        chosen = None
        chosen_source = None
        if fixed_result:
            chosen = fixed_result[0]
            chosen_source = "固定阈值"
        elif inv_result:
            chosen = inv_result[0]
            chosen_source = "反色"

        if debug:
            fixed_str = f"{fixed_result[0]}({fixed_result[1]:.2f})" if fixed_result else "失败"
            inv_str = f"{inv_result[0]}({inv_result[1]:.2f})" if inv_result else "失败"
            print(
                f"[部署栏OCR] {name} (bar[{bar_index}]): "
                f"固定阈值={fixed_str}, 反色={inv_str} → "
                f"{chosen if chosen is not None else '失败'}"
                f"{f' ({chosen_source})' if chosen is not None else ''}"
            )
            # 避免中文文件名导致保存异常，使用 bar 索引命名
            cv2.imwrite(os.path.join(session_dir, f"{bar_index:02d}_raw.png"), img)
            cv2.imwrite(os.path.join(session_dir, f"{bar_index:02d}_fixed.png"), fixed_img)
            cv2.imwrite(os.path.join(session_dir, f"{bar_index:02d}_inv.png"), inv_img)

        if chosen is not None:
            mapping[name] = chosen

    return mapping


def extract_cost_with_conf(results: list, min_conf: float = 0.5) -> Optional[Tuple[int, float]]:
    """从 OCR 结果中提取置信度最高的纯数字费用，无有效结果返回 None。"""
    best_cost = None
    best_conf = 0.0
    for bbox, (text, conf) in results:
        if conf < min_conf:
            continue
        digits = "".join(c for c in text if c.isdigit())
        if not digits:
            continue
        try:
            cost = int(digits)
        except ValueError:
            continue
        if conf > best_conf:
            best_conf = conf
            best_cost = cost
    return (best_cost, best_conf) if best_cost is not None else None


def preprocess_cost_image(img: np.ndarray) -> np.ndarray:
    """预处理费用数字截图：放大后固定阈值二值化并轻微闭运算，强化白字。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(
        gray,
        constants.DEPLOY_BAR_COST_WHITE_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def preprocess_cost_image_inv(img: np.ndarray) -> np.ndarray:
    """反色二值化（黑字白底），作为固定阈值失败时的回退。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
