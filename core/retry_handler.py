import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pydirectinput
from core.capture import WindowCapture
from core.stage_selector import StageSelector


class StageRetryHandler:
    """漏怪后的自动重试处理器。

    流程：
    1. ROI 区域模板匹配检测漏怪提示
    2. 点击返回/关闭提示
    3. 点击重新挑战（两次）
    4. 等待加载后再次点击
    5. 调用 StageSelector 重新进入关卡
    """

    # 比例坐标（基于 2560x1600）
    # ROI 区域: x1=1622, y1=26, x2=1715, y2=51 (基于 2560x1600)
    _LEAK_ROI_RATIO = (1622 / 2560, 26 / 1600, (1715 - 1622) / 2560, (51 - 26) / 1600)
    _CLICK_RETURN_RATIO = (131 / 2560, 73 / 1600)
    _CLICK_RETRY_RATIO = (1912 / 2560, 1194 / 1600)

    def __init__(
        self,
        capture: WindowCapture,
        selector: StageSelector,
        template_path: Optional[str] = None,
        threshold: float = 0.8,
        debug: bool = False,
    ):
        self.capture = capture
        self.selector = selector
        self.threshold = threshold
        self.debug = debug
        self._template: Optional[np.ndarray] = None
        if template_path:
            self.load_template(template_path)

    def load_template(self, path: str):
        # cv2.imread 对中文路径支持不佳，改用 imdecode
        try:
            self._template = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            self._template = None
        if self._template is None:
            raise FileNotFoundError(f"漏怪模板图片不存在或无法读取: {path}")
        if self._template.ndim == 3 and self._template.shape[2] == 3:
            self._template = cv2.cvtColor(self._template, cv2.COLOR_BGR2BGRA)

    def _ratio_to_pixel(self, rx: float, ry: float) -> Tuple[int, int]:
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return left + int(w * rx), top + int(h * ry)

    def _get_roi(self) -> Tuple[int, int, int, int]:
        """获取漏怪检测 ROI 的像素坐标 (x, y, w, h)，做边界保护。"""
        w, h = self.capture.get_window_size()
        rx, ry, rw, rh = self._LEAK_ROI_RATIO
        x = int(w * rx)
        y = int(h * ry)
        roi_w = int(w * rw)
        roi_h = int(h * rh)
        # 边界保护
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)
        return x, y, roi_w, roi_h

    def check_leak(self) -> bool:
        """在 ROI 区域执行模板匹配，返回是否检测到漏怪。"""
        if self._template is None:
            return False
        frame = self.capture.capture()
        x, y, roi_w, roi_h = self._get_roi()
        if roi_w <= 0 or roi_h <= 0:
            return False
        roi = frame[y : y + roi_h, x : x + roi_w]
        if roi.shape[0] < self._template.shape[0] or roi.shape[1] < self._template.shape[1]:
            if self.debug:
                print(f"[漏怪检测] ROI({roi_w}x{roi_h}) 小于模板({self._template.shape[1]}x{self._template.shape[0]})")
            return False
        result = cv2.matchTemplate(roi, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if self.debug:
            print(f"[漏怪检测] 匹配值={max_val:.3f}, 阈值={self.threshold}, ROI=({x},{y},{roi_w},{roi_h})")
        return max_val > self.threshold

    async def _perform_retry_clicks(self):
        """执行漏怪后的重试点击流程。"""
        # 1. 点击返回/关闭提示
        x, y = self._ratio_to_pixel(*self._CLICK_RETURN_RATIO)
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(2.0)

        # 2. 点击重新挑战
        x, y = self._ratio_to_pixel(*self._CLICK_RETRY_RATIO)
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(2.0)

        # 3. 再点击一次（确认）
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(10.0)

        # 4. 等待加载后再点击一次
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(2.0)

    async def run_retry_loop(
        self,
        stage_name: str,
        max_retries: int = 3,
        check_interval: float = 1.0,
    ) -> bool:
        """运行重试循环：检测漏怪 -> 重试点击 -> 重新选关。

        返回 True 表示在某次尝试中成功进入关卡（未检测到漏怪），
        False 表示重试用尽。
        """
        for attempt in range(1, max_retries + 1):
            print(f"[重试] 第 {attempt}/{max_retries} 次尝试...")

            # 先进入关卡
            ok = await self.selector.enter_stage(stage_name)
            if not ok:
                print("[重试] 进入关卡失败，跳过本次尝试")
                continue

            # 等待一段时间让关卡开始（给漏怪检测留出时间窗口）
            # 这里由外层控制，本方法只负责检测和重试
            # 实际上应该在关卡执行过程中检测漏怪
            # 但这里简化为：进入关卡后持续检测

            # 持续检测漏怪，如果检测到就重试
            # 注意：实际使用时应在 executor.run() 并行运行检测
            # 这里提供一个简化版本
            leak_detected = await self._wait_for_leak_or_timeout(check_interval)
            if not leak_detected:
                print("[重试] 本次尝试未检测到漏怪，成功")
                return True

            print("[重试] 检测到漏怪，执行重试流程...")
            await self._perform_retry_clicks()

        print("[重试] 重试次数已用完")
        return False

    async def _wait_for_leak_or_timeout(
        self, check_interval: float = 1.0, timeout: float = 300.0
    ) -> bool:
        """持续检测漏怪直到检测到或超时。"""
        end = time.time() + timeout
        while time.time() < end:
            if self.check_leak():
                return True
            await asyncio.sleep(check_interval)
        return False

    async def handle_leak_once(self, stage_name: str, should_stop=None) -> bool:
        """单次漏怪处理：执行重试点击并重新进入关卡。"""
        if should_stop is not None and should_stop():
            return False
        print("[漏怪处理] 检测到漏怪，开始重试流程...")
        await self._perform_retry_clicks()
        if should_stop is not None and should_stop():
            return False
        return await self.selector.enter_stage(stage_name, should_stop=should_stop)
