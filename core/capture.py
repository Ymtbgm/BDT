import mss
import numpy as np
import cv2
import win32gui
import win32ui
import win32con
import ctypes
from ctypes import windll
from typing import Optional, Tuple

# 开启 DPI Awareness，确保 GetClientRect 返回物理像素
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()


class WindowCapture:
    def __init__(self, window_title: str = "明日方舟", backend: str = "printwindow"):
        self.window_title = window_title
        self.backend = backend
        self.sct = mss.mss()
        self.monitor = None
        self._hwnd = None
        self._update_window_rect()

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
        screenshot = self.sct.grab(self.monitor)
        # 直接从 raw bytes 创建 numpy，避免 np.array(ScreenShot) 的内部打包开销
        img = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
            (screenshot.height, screenshot.width, 4)
        )
        return img

    def capture(self) -> np.ndarray:
        if self.backend == "printwindow":
            try:
                return self._capture_printwindow()
            except Exception as e:
                # 回退到 mss
                return self._capture_mss()
        return self._capture_mss()

    def get_window_size(self) -> Tuple[int, int]:
        if self.monitor is None:
            self._update_window_rect()
        return self.monitor["width"], self.monitor["height"]

    def capture_roi(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        """截取屏幕指定 ROI（绝对屏幕坐标），返回 BGRA。"""
        monitor = {"left": x, "top": y, "width": w, "height": h}
        screenshot = self.sct.grab(monitor)
        img = np.frombuffer(screenshot.raw, dtype=np.uint8).reshape(
            (screenshot.height, screenshot.width, 4)
        )
        return img

    def refresh_rect(self):
        self._update_window_rect()
