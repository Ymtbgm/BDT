from typing import List, Optional, Tuple
from pathlib import Path

import cv2
import numpy as np

from core.capture import WindowCapture
from core.cost_bar_calibration import CostBarCalibration, get_calibration
import core.constants as constants
from core.paths import game_template


class CostBarSyncCC:
    """费用条帧同步（基于校准表）。

    支持单校准表，也支持按时间切换多张校准表（例如普通模式前 10 秒与 10 秒后
    费用条第 29 帧行为不同）。通过最近邻匹配估算当前帧号；对存在 0 白像素歧义
    的阶段，会结合当前时间 target 帧号进行解歧义。
    """

    # 默认 ROI 比例基于 2560x1600 分辨率下费用条位置（与 main.py 中一致）
    DEFAULT_ROI_RATIOS = constants.COST_BAR_ROI_RATIOS

    def __init__(
        self,
        capture: WindowCapture,
        calibration_name: str,
        calibration_schedule: Optional[List[Tuple[float, str]]] = None,
        roi_ratios: Optional[Tuple[float, float, float, float]] = None,
        threshold: int = constants.COST_BAR_THRESHOLD,
        frame_offset_ms: float = constants.COST_BAR_FRAME_OFFSET_MS,
        debug: bool = False,
    ):
        self.capture = capture
        self.roi_ratios = roi_ratios or self.DEFAULT_ROI_RATIOS
        self.threshold = threshold
        self.frame_offset_ms = frame_offset_ms
        self.debug = debug

        if calibration_schedule is not None:
            self._schedule = sorted(calibration_schedule, key=lambda x: x[0])
            self._calibrations = {
                name: get_calibration(name) for _, name in self._schedule
            }
        else:
            self._schedule = None
            self._calibrations = {
                calibration_name: get_calibration(calibration_name)
            }
        self._default_calibration: CostBarCalibration = next(
            iter(self._calibrations.values())
        )

        # 加载费用 MAX 模板，用于检测费用条已满
        self._cost_max_template: Optional[np.ndarray] = None
        self._cost_max_mask: Optional[np.ndarray] = None
        cost_max_path = game_template(constants.COST_MAX_TEMPLATE_NAME)
        self._cost_max_template, self._cost_max_mask = self._load_template_with_mask(
            str(cost_max_path)
        )
        if self._cost_max_template is None and self.debug:
            print(f"[费用条同步-CC] 无法加载费用 MAX 模板: {cost_max_path}")

    @staticmethod
    def _load_template_with_mask(
        path: str,
        mask_threshold: int = constants.RATE_TEMPLATE_MASK_THRESHOLD,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """加载模板并生成前景掩膜（BGRA 按 alpha+亮度，BGR/灰度按亮度阈值）。"""
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None
            if img.ndim == 3 and img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                alpha = img[:, :, 3]
                mask = (
                    (alpha > mask_threshold) & (gray > mask_threshold)
                ).astype(np.uint8) * 255
            elif img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
            else:
                gray = img
                _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
            return gray, mask
        except Exception:
            return None, None

    def is_cost_max(self, roi_gray: Optional[np.ndarray] = None) -> bool:
        """检测费用条 ROI 是否出现 MAX 字样（费用已满）。"""
        if self._cost_max_template is None:
            return False
        img = roi_gray if roi_gray is not None else self.capture_roi_gray()
        if img is None:
            return False
        if (
            img.shape[0] < self._cost_max_template.shape[0]
            or img.shape[1] < self._cost_max_template.shape[1]
        ):
            return False
        try:
            result = cv2.matchTemplate(
                img, self._cost_max_template, cv2.TM_CCOEFF_NORMED, mask=self._cost_max_mask
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return max_val >= constants.COST_MAX_MATCH_CONFIDENCE
        except Exception:
            return False

    def get_calibration(self, time_ms: Optional[float] = None) -> CostBarCalibration:
        """根据时间选择当前生效的校准表。"""
        if self._schedule is None or time_ms is None:
            return self._default_calibration
        cal = self._calibrations[self._schedule[0][1]]
        for threshold, name in self._schedule:
            if time_ms >= threshold:
                cal = self._calibrations[name]
            else:
                break
        return cal

    @property
    def calibration(self) -> CostBarCalibration:
        return self._default_calibration

    @property
    def cycle_length(self) -> int:
        return self._default_calibration.cycle_length

    @property
    def frame_duration_ms(self) -> float:
        return self._default_calibration.frame_duration_ms

    def _roi_abs(self) -> Tuple[int, int, int, int]:
        """根据窗口大小计算费用条 ROI 的绝对屏幕坐标。"""
        w, h = self.capture.get_window_size()
        x = int(w * self.roi_ratios[0])
        y = int(h * self.roi_ratios[1])
        rw = int(w * self.roi_ratios[2])
        rh = int(h * self.roi_ratios[3])
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return left + x, top + y, rw, rh

    def capture_roi_gray(self) -> Optional[np.ndarray]:
        """截取费用条 ROI 并转为灰度图。"""
        try:
            x, y, w, h = self._roi_abs()
            img = self.capture.capture_roi(x, y, w, h)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return gray
        except Exception as e:
            if self.debug:
                print(f"[费用条同步-CC] 截取 ROI 失败: {e}")
            return None

    def white_pixel_count(self, roi_gray: Optional[np.ndarray] = None) -> Optional[int]:
        """统计 ROI 内白像素（灰度 > threshold）数量。"""
        img = roi_gray if roi_gray is not None else self.capture_roi_gray()
        if img is None:
            return None
        return int(np.sum(img > self.threshold))

    def expected_count(self, frame_index: int, time_ms: Optional[float] = None) -> int:
        """返回指定帧号的期望白像素数量。"""
        cal = self.get_calibration(time_ms)
        frame_index = frame_index % cal.cycle_length
        return cal.expected_counts[frame_index]

    def is_match(
        self,
        count: int,
        frame_index: int,
        time_ms: Optional[float] = None,
        tolerance: Optional[float] = None,
    ) -> bool:
        """判断白像素数量是否匹配指定帧号。"""
        cal = self.get_calibration(time_ms)
        expected = self.expected_count(frame_index, time_ms)
        if tolerance is None:
            prev_expected = self.expected_count(
                (frame_index - 1) % cal.cycle_length, time_ms
            )
            next_expected = self.expected_count(
                (frame_index + 1) % cal.cycle_length, time_ms
            )
            gaps = [abs(expected - prev_expected), abs(next_expected - expected)]
            nonzero_gaps = [g for g in gaps if g > 0]
            min_gap = min(nonzero_gaps) if nonzero_gaps else 30.0
            tolerance = max(5.0, min_gap * 0.45)
        return abs(count - expected) <= tolerance

    def target_frame_index(self, time_ms: float) -> int:
        """根据脚本实际时间计算费用条目标帧号。

        先把时间换算为游戏逻辑帧（30fps），使用四舍五入而不是向下取整，
        再对费用条更新周期取余，得到当前费用条应处的帧索引。
        """
        cal = self.get_calibration(time_ms)
        adjusted = max(0.0, time_ms - self.frame_offset_ms)
        logical_frame = int(30.0 * adjusted / 1000.0 + 0.5)
        return logical_frame % cal.cycle_length

    def current_frame(
        self, count: Optional[int] = None, time_ms: Optional[float] = None
    ) -> Optional[int]:
        """根据白像素数量估算当前帧号，返回期望白像素最接近的帧索引。

        若当前校准表存在多个 0 白像素帧（如 normal_early 第 0/29 帧，或
        危机合约 tag 循环末尾与起始帧均为 0），会结合 target_frame_index
        优先返回与当前时间一致的帧号。当计时器略快于费用条、target 已跳到
        非零帧但白像素仍在 0 帧区域时，也会优先跟随 target，避免回退到 frame 0。
        """
        if count is None:
            count = self.white_pixel_count()
        if count is None:
            return None

        cal = self.get_calibration(time_ms)
        expected = cal.expected_counts
        best_idx = 0
        best_diff = abs(expected[0] - count)
        for i in range(1, len(expected)):
            diff = abs(expected[i] - count)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        zero_indices = [i for i, c in enumerate(expected) if c == 0]
        has_zero_ambiguity = len(zero_indices) > 1
        within_time_window = (
            cal.zero_ambiguous_until_ms is not None
            and time_ms is not None
            and time_ms < cal.zero_ambiguous_until_ms
        )

        if (has_zero_ambiguity or within_time_window) and time_ms is not None:
            target = self.target_frame_index(time_ms)
            # 只在“空帧”区域解歧义，避免把非空帧误判到另一帧
            if expected[target] == 0 and expected[target] == expected[best_idx]:
                return target
            # 计时器可能略快于游戏，target 已跳到非零帧但 count 还在 0 帧区域。
            # 此时先尝试跟随 target（count 已走到目标帧一半以上）；
            # 否则返回离 target 最近的 0 帧，避免直接回退到 frame 0。
            if expected[best_idx] == 0 and expected[target] > 0:
                if abs(count - expected[target]) <= abs(count - expected[best_idx]):
                    return target
                closest_zero = min(
                    zero_indices,
                    key=lambda i: self.frame_distance(i, target, time_ms),
                )
                return closest_zero

        return best_idx

    def frame_distance(self, a: int, b: int, time_ms: Optional[float] = None) -> int:
        """计算两个循环帧号之间的最短距离。"""
        cycle = self.get_calibration(time_ms).cycle_length
        d = abs(a - b)
        return min(d, cycle - d)

    def debug_info(self, time_ms: float) -> dict:
        """返回当前帧同步的调试信息。"""
        count = self.white_pixel_count()
        target = self.target_frame_index(time_ms)
        current = self.current_frame(count, time_ms)
        cal = self.get_calibration(time_ms)
        return {
            "white_count": count,
            "current_frame": current,
            "target_frame": target,
            "frame_distance": self.frame_distance(current, target, time_ms) if current is not None else None,
            "target_match": self.is_match(count, target, time_ms) if count is not None else None,
            "next_match": self.is_match(count, (target + 1) % cal.cycle_length, time_ms) if count is not None else None,
        }
