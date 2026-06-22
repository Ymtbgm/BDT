import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from core.capture import WindowCapture


class CostBarStartDetector:
    """基于 COST 图标 + 费用条变化的启动检测器。

    复用 main.py 中 _wait_for_game_start 的逻辑，提供非阻塞 tick() 和阻塞式 detect_async()。
    """

    STATE_WAIT_COST = "wait_cost"
    STATE_WAIT_37FRAMES = "wait_37frames"
    STATE_WAIT_BAR = "wait_bar"
    STATE_DONE = "done"

    # 基于 2560x1600 的比例坐标
    _COST_X_RATIO = 2355 / 2560
    _COST_X2_RATIO = 2412 / 2560
    _COST_Y_RATIO = 1208 / 1600
    _COST_Y2_RATIO = 1264 / 1600
    _BAR_X_RATIO = 2343 / 2560
    _BAR_X2_RATIO = 2560 / 2560
    _BAR_Y_RATIO = 1278 / 1600
    _BAR_Y2_RATIO = 1284 / 1600

    def __init__(
        self,
        capture: WindowCapture,
        cost_template: np.ndarray,
        debug: bool = False,
    ):
        self.capture = capture
        self.cost_template = cost_template
        self.debug = debug
        self._state = self.STATE_WAIT_COST
        self._wait_37frames_until: float = 0.0
        self._bar_start_time: float = 0.0
        self._bar_prev_roi: Optional[np.ndarray] = None
        self._cost_roi: Optional[Tuple[int, int, int, int]] = None
        self._bar_roi: Optional[Tuple[int, int, int, int]] = None
        self._refresh_rois()

    def _refresh_rois(self):
        """根据当前窗口位置刷新 ROI 绝对坐标。"""
        win_left = self.capture.monitor.get("left", 0)
        win_top = self.capture.monitor.get("top", 0)
        w, h = self.capture.get_window_size()

        cost_x = win_left + int(w * self._COST_X_RATIO)
        cost_y = win_top + int(h * self._COST_Y_RATIO)
        cost_w = int(w * self._COST_X2_RATIO) - int(w * self._COST_X_RATIO)
        cost_h = int(h * self._COST_Y2_RATIO) - int(h * self._COST_Y_RATIO)
        self._cost_roi = (cost_x, cost_y, cost_w, cost_h)

        bar_x = win_left + int(w * self._BAR_X_RATIO)
        bar_y = win_top + int(h * self._BAR_Y_RATIO)
        bar_w = int(w * self._BAR_X2_RATIO) - int(w * self._BAR_X_RATIO)
        bar_h = int(h * self._BAR_Y2_RATIO) - int(h * self._BAR_Y_RATIO)
        self._bar_roi = (bar_x, bar_y, bar_w, bar_h)

    def reset(self):
        """重置到初始等待 COST 图标状态。"""
        self._state = self.STATE_WAIT_COST
        self._wait_37frames_until = 0.0
        self._bar_start_time = 0.0
        self._bar_prev_roi = None
        self._refresh_rois()

    @property
    def state(self) -> str:
        return self._state

    def tick(
        self,
        cost_threshold: float = 0.8,
        bar_timeout: float = 10.0,
        diff_threshold: float = 3.0,
    ) -> Optional[float]:
        """非阻塞推进一帧检测。返回 offset_ms 表示完成，返回 None 表示继续等待。"""
        if self._state == self.STATE_DONE:
            return 0.0

        if self._state == self.STATE_WAIT_COST:
            return self._tick_wait_cost(cost_threshold)
        if self._state == self.STATE_WAIT_37FRAMES:
            return self._tick_wait_37frames()
        if self._state == self.STATE_WAIT_BAR:
            return self._tick_wait_bar(bar_timeout, diff_threshold)
        return None

    def _tick_wait_cost(self, cost_threshold: float) -> Optional[float]:
        if self._cost_roi is None:
            return None
        x, y, w, h = self._cost_roi
        try:
            roi = self.capture.capture_roi(x, y, w, h)
            if roi.size == 0:
                return None
            result = cv2.matchTemplate(roi, self.cost_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            if self.debug:
                print(f"[DEBUG] COST 匹配值: {max_val:.3f}")
            if max_val >= cost_threshold:
                if self.debug:
                    print(f"[计时校准] COST 图标已检测到 (置信度: {max_val:.3f})")
                self._state = self.STATE_WAIT_37FRAMES
                self._wait_37frames_until = time.perf_counter() + 37 / 30.0
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] COST 检测异常: {e}")
        return None

    def _tick_wait_37frames(self) -> Optional[float]:
        if time.perf_counter() < self._wait_37frames_until:
            return None
        if self.debug:
            wait_ms = 37 / 30.0 * 1000
            print(f"[计时校准] 等待 {wait_ms:.1f}ms (37帧) 跳过画面渐入...")
        self._state = self.STATE_WAIT_BAR
        self._bar_start_time = time.perf_counter()
        self._bar_prev_roi = None
        return None

    def _tick_wait_bar(
        self, bar_timeout: float, diff_threshold: float
    ) -> Optional[float]:
        if self._bar_roi is None:
            return None

        if time.perf_counter() - self._bar_start_time > bar_timeout:
            if self.debug:
                print("[计时校准] 费用条检测超时")
            self._state = self.STATE_DONE
            return 0.0

        x, y, w, h = self._bar_roi
        try:
            roi = self.capture.capture_roi(x, y, w, h)
            if roi.size == 0:
                return None

            if self._bar_prev_roi is not None and self._bar_prev_roi.shape == roi.shape:
                diff = cv2.absdiff(roi, self._bar_prev_roi)
                mean_diff = float(np.mean(diff))
                if self.debug:
                    print(f"[DEBUG] 费用条 差分={mean_diff:.2f}")
                if mean_diff > diff_threshold:
                    offset_ms = 2 * 1000.0 / 30.0 - 15.0
                    if self.debug:
                        print(
                            f"[计时校准] 费用条变化 detected (差分: {mean_diff:.2f}, "
                            f"修正: -{offset_ms:.1f}ms)"
                        )
                        debug_dir = Path("debug")
                        debug_dir.mkdir(exist_ok=True)
                        ts = int(time.perf_counter() * 1000)
                        cv2.imencode(".png", roi)[1].tofile(
                            str(debug_dir / f"bar_trigger_{ts}.png")
                        )
                        if self._bar_prev_roi is not None:
                            cv2.imencode(".png", self._bar_prev_roi)[1].tofile(
                                str(debug_dir / f"bar_prev_{ts}.png")
                            )
                        print(
                            f"[DEBUG] 已保存费用条截图: debug/bar_trigger_{ts}.png 与 bar_prev_{ts}.png"
                        )
                    self._state = self.STATE_DONE
                    return offset_ms

            self._bar_prev_roi = roi.copy()
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] 费用条检测异常: {e}")
        return None

    async def detect_async(
        self,
        cost_threshold: float = 0.8,
        cost_timeout: float = 30.0,
        bar_timeout: float = 10.0,
        interval: float = 0.01,
        should_stop=None,
    ) -> float:
        """异步阻塞式检测，返回 offset_ms。

        should_stop: 可选的无参可调用对象，返回 True 时立即终止检测并返回 0.0。
        """
        import asyncio

        start = time.perf_counter()
        while self._state == self.STATE_WAIT_COST:
            if should_stop is not None and should_stop():
                return 0.0
            if time.perf_counter() - start > cost_timeout:
                if self.debug:
                    print("[计时校准] 等待 COST 图标超时")
                self._state = self.STATE_DONE
                return 0.0
            result = self.tick(cost_threshold, bar_timeout)
            if result is not None:
                return result
            await asyncio.sleep(interval)

        # 37 帧等待 + 费用条检测
        while self._state != self.STATE_DONE:
            if should_stop is not None and should_stop():
                return 0.0
            result = self.tick(cost_threshold, bar_timeout)
            if result is not None:
                return result
            await asyncio.sleep(interval)

        return 0.0

    @staticmethod
    def load_template(path: str) -> Optional[np.ndarray]:
        """加载 COST 模板，支持 BGR/BGRA。"""
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            return img
        except Exception:
            return None

    @staticmethod
    def default_template_path(root: Optional[Path] = None) -> Path:
        """返回默认 cost.png 路径。"""
        if root is None:
            import sys

            root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent.parent
        return root / "core" / "resource" / "cost.png"
