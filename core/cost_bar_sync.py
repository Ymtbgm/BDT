from typing import Optional, Tuple

import cv2
import numpy as np

from core.capture import WindowCapture
import core.constants as constants


class CostBarSync:
    """基于费用条白像素数量进行帧级同步。

    费用条以约 45 像素/帧的速度填充，一帧为 33ms。
    同步策略：
      1. 若当前白像素已匹配目标帧或下一帧，直接执行；
      2. 否则调用 p_and_esc_click 跳 1 帧，再次匹配目标帧；
      3. 仍未匹配则再跳 1 帧并强制执行（累计误差较大时兜底）。
    """

    # 默认 ROI 比例基于 2560x1600 分辨率下费用条位置（与 main.py 中一致）
    DEFAULT_ROI_RATIOS = constants.COST_BAR_ROI_RATIOS

    def __init__(
        self,
        capture: WindowCapture,
        roi_ratios: Optional[Tuple[float, float, float, float]] = None,
        threshold: int = constants.COST_BAR_THRESHOLD,
        step_pixels: float = constants.COST_BAR_STEP_PIXELS,
        full_pixels: int = constants.COST_BAR_FULL_PIXELS,
        frame_offset_ms: float = constants.COST_BAR_FRAME_OFFSET_MS,
        debug: bool = False,
    ):

        self.capture = capture
        self.roi_ratios = roi_ratios or self.DEFAULT_ROI_RATIOS
        self.threshold = threshold
        self.step_pixels = step_pixels
        self.full_pixels = full_pixels
        self.frame_offset_ms = frame_offset_ms
        self.debug = debug

    @property
    def cycle_length(self) -> int:
        return 30

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
                print(f"[费用条同步] 截取 ROI 失败: {e}")
            return None

    def white_pixel_count(self, roi_gray: Optional[np.ndarray] = None) -> Optional[int]:
        """统计 ROI 内白像素（灰度 > threshold）数量。"""
        img = roi_gray or self.capture_roi_gray()
        if img is None:
            return None
        return int(np.sum(img > self.threshold))

    def expected_count(self, frame_index: int) -> int:
        """计算指定帧号的期望白像素数量。"""
        if frame_index == 29:
            # 尾帧可能是未满的 1254 或满的 1302，统一按满条处理
            return self.full_pixels
        return int(frame_index * self.step_pixels)

    def is_match(self, count: int, frame_index: int, tolerance: Optional[float] = None) -> bool:
        """判断白像素数量是否匹配指定帧号。"""
        if tolerance is None:
            tolerance = self.step_pixels * 0.7
        expected = self.expected_count(frame_index)
        if frame_index == 29:
            # 尾帧：只要接近满条即视为 29 帧
            return count >= expected - tolerance
        return abs(count - expected) <= tolerance

    def target_frame_index(self, time_ms: float) -> int:
        """根据理论时间计算目标帧号（0-29）。

        先取秒内余数再除以 33.3ms，避免用 33ms 直接除 1000ms 造成的累计帧误差。
        """
        adjusted = max(0.0, time_ms - self.frame_offset_ms)
        return int((adjusted % 1000) // 33.3) % 30

    def current_frame(self, count: Optional[int] = None) -> Optional[int]:
        """根据白像素数量估算当前帧号（0-29），按最近理论帧像素值四舍五入。"""
        if count is None:
            count = self.white_pixel_count()
        if count is None:
            return None

        # 接近满条时直接判为尾帧
        tail_threshold = self.full_pixels - self.step_pixels * 1.1
        if count >= tail_threshold:
            return 29

        base = int(count // self.step_pixels)
        rem = int(count % self.step_pixels)
        frame = base + (1 if rem > self.step_pixels / 2 else 0)
        return max(0, min(29, frame))

    @staticmethod
    def frame_distance(a: int, b: int) -> int:
        """计算两个 30 帧周期帧号之间的最短距离。"""
        d = abs(a - b)
        return min(d, 30 - d)

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
            "next_match": self.is_match(count, (target + 1) % 30) if count is not None else None,
        }
