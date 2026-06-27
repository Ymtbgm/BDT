from typing import Optional, Tuple

import cv2
import numpy as np

from core.capture import WindowCapture
from core.cost_bar_calibration import CostBarCalibration, get_calibration
import core.constants as constants


class CostBarSyncCC:
    """危机合约费用条帧同步（基于校准表）。

    与常规 CostBarSync 不同，危机合约 tag 会改变费用条每帧白像素的分布，
    不再满足简单的线性增长。因此使用预先测量的校准表，通过最近邻匹配来
    估算当前帧号。
    """

    # 默认 ROI 比例基于 2560x1600 分辨率下费用条位置（与 main.py 中一致）
    DEFAULT_ROI_RATIOS = constants.COST_BAR_ROI_RATIOS

    def __init__(
        self,
        capture: WindowCapture,
        calibration_name: str,
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
        self._calibration: CostBarCalibration = get_calibration(calibration_name)

    @property
    def calibration(self) -> CostBarCalibration:
        return self._calibration

    @property
    def cycle_length(self) -> int:
        return self._calibration.cycle_length

    @property
    def frame_duration_ms(self) -> float:
        return self._calibration.frame_duration_ms

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
        img = roi_gray or self.capture_roi_gray()
        if img is None:
            return None
        return int(np.sum(img > self.threshold))

    def expected_count(self, frame_index: int) -> int:
        """返回指定帧号的期望白像素数量。"""
        frame_index = frame_index % self.cycle_length
        return self._calibration.expected_counts[frame_index]

    def is_match(self, count: int, frame_index: int, tolerance: Optional[float] = None) -> bool:
        """判断白像素数量是否匹配指定帧号。"""
        expected = self.expected_count(frame_index)
        if tolerance is None:
            prev_expected = self.expected_count((frame_index - 1) % self.cycle_length)
            next_expected = self.expected_count((frame_index + 1) % self.cycle_length)
            gaps = [abs(expected - prev_expected), abs(next_expected - expected)]
            nonzero_gaps = [g for g in gaps if g > 0]
            min_gap = min(nonzero_gaps) if nonzero_gaps else 30.0
            tolerance = max(5.0, min_gap * 0.45)
        return abs(count - expected) <= tolerance

    def target_frame_index(self, time_ms: float) -> int:
        """根据脚本实际时间计算费用条目标帧号。

        先把时间换算为游戏逻辑帧（30fps），再对费用条更新周期取余，
        得到当前费用条应处的帧索引。
        """
        adjusted = max(0.0, time_ms - self.frame_offset_ms)
        logical_frame = int(30.0 * adjusted / 1000.0)
        return logical_frame % self.cycle_length

    def current_frame(self, count: Optional[int] = None) -> Optional[int]:
        """根据白像素数量估算当前帧号，返回期望白像素最接近的帧索引。"""
        if count is None:
            count = self.white_pixel_count()
        if count is None:
            return None

        expected = self._calibration.expected_counts
        best_idx = 0
        best_diff = abs(expected[0] - count)
        for i in range(1, len(expected)):
            diff = abs(expected[i] - count)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def frame_distance(self, a: int, b: int) -> int:
        """计算两个循环帧号之间的最短距离。"""
        cycle = self.cycle_length
        d = abs(a - b)
        return min(d, cycle - d)

    def debug_info(self, time_ms: float) -> dict:
        """返回当前帧同步的调试信息。"""
        count = self.white_pixel_count()
        target = self.target_frame_index(time_ms)
        current = self.current_frame(count)
        return {
            "white_count": count,
            "current_frame": current,
            "target_frame": target,
            "frame_distance": self.frame_distance(current, target) if current is not None else None,
            "target_match": self.is_match(count, target) if count is not None else None,
            "next_match": self.is_match(count, (target + 1) % self.cycle_length) if count is not None else None,
        }
