import time
import cv2
import numpy as np
from typing import Optional, Callable
from .capture import WindowCapture


class LeakDetector:
    def __init__(self, capture: WindowCapture):
        self.capture = capture
        self._running = False
        self._callbacks: list[Callable] = []
        self._template: Optional[np.ndarray] = None
        self._check_interval = 1.0

    def set_template(self, template_path: str):
        try:
            self._template = cv2.imdecode(np.fromfile(template_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            self._template = None
        if self._template is None:
            raise FileNotFoundError(f"找不到模板图片: {template_path}")
        if self._template.ndim == 3 and self._template.shape[2] == 3:
            self._template = cv2.cvtColor(self._template, cv2.COLOR_BGR2BGRA)

    def register_leak_callback(self, callback: Callable):
        self._callbacks.append(callback)

    def _detect_by_template(self, frame: np.ndarray) -> bool:
        if self._template is None:
            return False
        result = cv2.matchTemplate(frame, self._template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return max_val > 0.8

    def check_once(self) -> bool:
        frame = self.capture.capture()
        if self._template is not None and self._detect_by_template(frame):
            return True
        return False

    async def start_monitoring(self, interval: float = 5.0):
        import asyncio
        self._running = True
        self._check_interval = interval
        while self._running:
            if self.check_once():
                for cb in self._callbacks:
                    cb()
                break
            await asyncio.sleep(self._check_interval)

    def stop(self):
        self._running = False
