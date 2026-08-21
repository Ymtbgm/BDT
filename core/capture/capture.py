from __future__ import annotations

import atexit
import mss
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
import ctypes
import threading
import time
from collections import deque
from ctypes import windll
from typing import Optional, Tuple

# 开启 DPI Awareness，确保 GetClientRect 返回物理像素
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl, CaptureControl
    _WINDOWS_CAPTURE_AVAILABLE = True
except Exception:
    _WINDOWS_CAPTURE_AVAILABLE = False


class _WindowsCaptureBackend:
    """基于 Windows Graphics Capture 的异步截图后端。

    通过 window_hwnd 绑定目标窗口，在独立线程持续捕获最新帧并缓存；
    capture_roi 从缓存帧中裁剪，耗时通常 <0.1ms。
    """

    def __init__(
        self,
        window_title: str,
        minimum_update_interval: int = 6,
        max_first_frame_wait_ms: float = 2000.0,
    ):
        self._window_title = window_title
        self._minimum_update_interval = minimum_update_interval
        self._max_first_frame_wait_ms = max_first_frame_wait_ms
        self._hwnd: Optional[int] = None
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_shape: Optional[Tuple[int, int]] = None
        self._control: Optional[CaptureControl] = None
        self._capture: Optional[WindowsCapture] = None
        self._monitor: Optional[dict] = None
        self._started = False
        self._closed = False
        # 记录最近 2s 内的帧到达时间戳，用于统计实际捕获帧率
        self._frame_arrival_times: deque = deque()
        self._arrival_lock = threading.Lock()

        self._start()
        atexit.register(self.stop)

    def _record_frame_arrival(self):
        now = time.perf_counter()
        with self._arrival_lock:
            self._frame_arrival_times.append(now)
            cutoff = now - 2.0
            while self._frame_arrival_times and self._frame_arrival_times[0] < cutoff:
                self._frame_arrival_times.popleft()

    def get_frame_arrival_intervals(self) -> list:
        """返回最近 2s 内相邻帧到达间隔（ms）。"""
        with self._arrival_lock:
            times = list(self._frame_arrival_times)
        if len(times) < 2:
            return []
        return [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]

    def _find_hwnd(self) -> int:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            return self._hwnd
        hwnd = win32gui.FindWindow(None, self._window_title)
        if hwnd == 0:
            hwnd = win32gui.FindWindow(None, "Arknights")
        if hwnd == 0:
            raise RuntimeError(f"找不到窗口: {self._window_title}")
        self._hwnd = hwnd
        return hwnd

    def _update_window_rect(self):
        hwnd = self._find_hwnd()
        rect = win32gui.GetClientRect(hwnd)
        left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
        right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
        self._monitor = {
            "left": left,
            "top": top,
            "width": right - left,
            "height": bottom - top,
        }

    def on_frame_arrived(self, frame: Frame, capture_control: InternalCaptureControl):
        self._record_frame_arrival()
        with self._lock:
            self._latest_frame = frame.frame_buffer.copy()
            self._frame_shape = self._latest_frame.shape

    def on_closed(self):
        self._closed = True

    def _start(self):
        if self._started:
            return
        self._update_window_rect()
        hwnd = self._find_hwnd()
        self._capture = WindowsCapture(
            cursor_capture=None,
            draw_border=False,
            minimum_update_interval=self._minimum_update_interval,
            window_hwnd=hwnd,
        )
        self._capture.event(self.on_frame_arrived)
        self._capture.event(self.on_closed)
        self._control = self._capture.start_free_threaded()
        self._started = True
        # 等待首帧
        deadline = time.perf_counter() + self._max_first_frame_wait_ms / 1000.0
        while time.perf_counter() < deadline:
            with self._lock:
                if self._latest_frame is not None:
                    return
            time.sleep(0.001)
        raise RuntimeError("windows-capture 未能在超时内获取首帧")

    def get_monitor(self) -> dict:
        if self._monitor is None:
            self._update_window_rect()
        return self._monitor

    def get_window_size(self) -> Tuple[int, int]:
        with self._lock:
            if self._frame_shape is not None:
                return self._frame_shape[1], self._frame_shape[0]
        # 回退到 win32 查询
        self._update_window_rect()
        return self._monitor["width"], self._monitor["height"]

    def capture(self) -> np.ndarray:
        with self._lock:
            if self._latest_frame is not None:
                return self._latest_frame.copy()
        raise RuntimeError("windows-capture 暂无可用帧")

    def capture_roi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """输入为绝对屏幕坐标，内部转换为窗口相对坐标后裁剪。"""
        self._update_window_rect()
        left = self._monitor["left"]
        top = self._monitor["top"]
        rel_x = x - left
        rel_y = y - top

        with self._lock:
            if self._latest_frame is None:
                raise RuntimeError("windows-capture 暂无可用帧")
            frame_h, frame_w = self._latest_frame.shape[:2]
            # 限制在帧范围内
            x0 = max(0, min(rel_x, frame_w))
            y0 = max(0, min(rel_y, frame_h))
            x1 = max(0, min(rel_x + w, frame_w))
            y1 = max(0, min(rel_y + h, frame_h))
            if x1 <= x0 or y1 <= y0:
                raise RuntimeError(
                    f"ROI ({x},{y},{w},{h}) 不在窗口 ({left},{top},{frame_w},{frame_h}) 内"
                )
            return self._latest_frame[y0:y1, x0:x1].copy()

    def refresh_rect(self):
        self._update_window_rect()

    def stop(self):
        if not self._started:
            return
        try:
            if self._control is not None:
                self._control.stop()
                self._control.wait()
        except Exception:
            pass
        self._started = False
        self._latest_frame = None


class WindowCapture:
    def __init__(
        self,
        window_title: str = "明日方舟",
        backend: str = "printwindow",
        minimum_update_interval: int = 6,
        debug: bool = False,
    ):
        self.window_title = window_title
        self.backend = backend
        self.debug = debug
        # mss 的 Windows DC 句柄存储在线程本地变量中，跨线程使用会报
        # "'_thread._local' object has no attribute 'srcdc'"，因此每个线程
        # 需要独立的 mss 实例。
        self._mss_local = threading.local()
        self._wgc_backend: Optional[_WindowsCaptureBackend] = None
        self.monitor = None
        self._hwnd = None

        if backend == "windows_capture":
            if not _WINDOWS_CAPTURE_AVAILABLE:
                raise RuntimeError("未安装 windows-capture 包，无法使用该后端")
            self._wgc_backend = _WindowsCaptureBackend(
                window_title=window_title,
                minimum_update_interval=minimum_update_interval,
            )
            self.monitor = self._wgc_backend.get_monitor()
        else:
            # 预热主线程的 mss 实例，同时完成窗口矩形初始化
            _ = self._get_mss()
            self._update_window_rect()
            if self.debug:
                print(f"[WindowCapture] backend={backend}, monitor={self.monitor}")

    def _get_mss(self):
        """返回当前线程的 mss 实例，按需创建。"""
        sct = getattr(self._mss_local, "sct", None)
        if sct is None:
            # 使用 mss.mss() 以保持最大兼容性；mss.MSS() 在部分版本/导入方式下可能不可用。
            sct = mss.mss()
            self._mss_local.sct = sct
        return sct

    def _find_hwnd(self) -> int:
        if self._hwnd is not None and win32gui.IsWindow(self._hwnd):
            return self._hwnd
        hwnd = win32gui.FindWindow(None, self.window_title)
        if hwnd == 0:
            hwnd = win32gui.FindWindow(None, "Arknights")
        if hwnd == 0:
            raise RuntimeError(f"找不到窗口: {self.window_title}")
        self._hwnd = hwnd
        return hwnd

    def _update_window_rect(self):
        try:
            hwnd = self._find_hwnd()
            rect = win32gui.GetClientRect(hwnd)
            left, top = win32gui.ClientToScreen(hwnd, (rect[0], rect[1]))
            right, bottom = win32gui.ClientToScreen(hwnd, (rect[2], rect[3]))
            self.monitor = {
                "left": left,
                "top": top,
                "width": right - left,
                "height": bottom - top,
            }
            if self.debug:
                print(f"[WindowCapture] _update_window_rect: hwnd={hwnd}, monitor={self.monitor}")
        except Exception as e:
            raise RuntimeError(f"获取窗口位置失败: {e}")

    def _capture_printwindow(self) -> np.ndarray:
        """使用 PrintWindow API 截取窗口客户区（支持后台/遮挡窗口）。"""
        hwnd = self._find_hwnd()
        left, top, right, bottom = win32gui.GetClientRect(hwnd)
        width, height = right - left, bottom - top

        hwndDC = win32gui.GetWindowDC(hwnd)
        mfcDC = win32ui.CreateDCFromHandle(hwndDC)
        saveDC = mfcDC.CreateCompatibleDC()

        saveBitMap = win32ui.CreateBitmap()
        saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
        saveDC.SelectObject(saveBitMap)

        # 3 = PW_CLIENTONLY | PW_RENDERFULLCONTENT
        result = windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 3)

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        img = np.frombuffer(bmpstr, dtype=np.uint8)

        if bmpinfo['bmBitsPixel'] == 32:
            img = img.reshape(height, width, 4)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            img = img.reshape(height, width, 3)

        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)

        if result == 0:
            raise RuntimeError("PrintWindow 截取失败")

        # 简单黑屏检测：如果全黑则抛异常让外层回退
        if np.mean(img) < 1.0:
            raise RuntimeError("PrintWindow 截取到黑屏")

        return img

    def _capture_mss(self) -> np.ndarray:
        """使用 mss 截取屏幕区域（前台截图）。"""
        if self.monitor is None:
            self._update_window_rect()
        screenshot = self._get_mss().grab(self.monitor)
        # 直接从 raw bytes 创建 numpy，避免 np.array(ScreenShot) 的内部打包开销
        img = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
            (screenshot.height, screenshot.width, 4)
        )
        return img

    def capture_mss(self) -> np.ndarray:
        """使用 mss 截取窗口客户区（前台截图，获取当前实际显示像素）。"""
        if self.monitor is None:
            self._update_window_rect()
        screenshot = self._get_mss().grab(self.monitor)
        img = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
            (screenshot.height, screenshot.width, 4)
        )
        return img

    def capture(self) -> np.ndarray:
        if self._wgc_backend is not None:
            return self._wgc_backend.capture()
        if self.backend == "printwindow":
            try:
                img = self._capture_printwindow()
                if self.debug:
                    print(f"[WindowCapture] capture(printwindow): shape={img.shape}, mean={img.mean():.1f}")
                return img
            except Exception as e:
                if self.debug:
                    print(f"[WindowCapture] printwindow failed: {e}, fallback to mss")
                # 回退到 mss
                return self._capture_mss()
        img = self._capture_mss()
        if self.debug:
            print(f"[WindowCapture] capture(mss): shape={img.shape}, mean={img.mean():.1f}")
        return img

    def get_window_size(self) -> Tuple[int, int]:
        if self._wgc_backend is not None:
            return self._wgc_backend.get_window_size()
        if self.monitor is None:
            self._update_window_rect()
        return self.monitor["width"], self.monitor["height"]

    def capture_roi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """截取屏幕指定 ROI（绝对屏幕坐标），返回 BGRA。"""
        if self._wgc_backend is not None:
            return self._wgc_backend.capture_roi(x, y, w, h)
        monitor = {"left": x, "top": y, "width": w, "height": h}
        try:
            screenshot = self._get_mss().grab(monitor)
            img = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
                (screenshot.height, screenshot.width, 4)
            )
            if self.debug:
                print(f"[WindowCapture] capture_roi: monitor={monitor}, shape={img.shape}, mean={img.mean():.1f}")
            return img
        except Exception as e:
            if self.debug:
                print(f"[WindowCapture] capture_roi FAILED: monitor={monitor}, error={e}")
            raise

    def refresh_rect(self):
        if self._wgc_backend is not None:
            self._wgc_backend.refresh_rect()
            self.monitor = self._wgc_backend.get_monitor()
        else:
            self._update_window_rect()

    def stop(self):
        """停止底层捕获资源（主要用于 windows-capture 后端）。"""
        if self._wgc_backend is not None:
            self._wgc_backend.stop()
        sct = getattr(self._mss_local, "sct", None)
        if sct is not None:
            try:
                sct.close()
            except Exception:
                pass
            self._mss_local.sct = None

    def get_frame_arrival_intervals(self) -> list:
        """返回 windows-capture 后端最近 2s 内的帧到达间隔（ms）；mss 后端返回空列表。"""
        if self._wgc_backend is not None:
            return self._wgc_backend.get_frame_arrival_intervals()
        return []
