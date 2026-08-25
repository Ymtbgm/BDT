"""Windows 窗口特效辅助函数（用于消除 Qt 6 悬浮窗的玻璃/边框残留）。"""

import sys

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes

    _dwmapi = ctypes.WinDLL("dwmapi")
    _DwmSetWindowAttribute = _dwmapi.DwmSetWindowAttribute

    # 与参考建议完全一致：只禁用系统背景材质
    DWMWA_SYSTEMBACKDROP_TYPE = 38
    DWMSBT_NONE = 1
    # Windows 11 圆角偏好
    DWMWA_WINDOW_CORNER_PREFERENCE = 33
    DWMWCP_DEFAULT = 0
    DWMWCP_DONOTROUND = 1
    DWMWCP_ROUND = 2
    DWMWCP_ROUNDSMALL = 3


def remove_dwm_glass_border(widget) -> bool:
    """
    关闭 Windows 11 DWM 在 frameless 透明窗口上产生的 Mica/Acrylic 背景
    以及系统自动添加的圆角，避免透明边框/玻璃感。
    必须在窗口 show() 之后调用，否则 winId 无效。
    """
    if sys.platform != "win32":
        print("[remove_dwm_glass_border] not on win32, skip")
        return False

    try:
        hwnd = int(widget.winId())

        # 1. 禁用系统背景材质
        value = ctypes.c_int(DWMSBT_NONE)
        result1 = _DwmSetWindowAttribute(
            hwnd,
            DWMWA_SYSTEMBACKDROP_TYPE,
            ctypes.byref(value),
            ctypes.sizeof(value),
        )

        # 2. 禁用 Windows 11 自动圆角，消除圆角边缘的半透明/玻璃感
        value2 = ctypes.c_int(DWMWCP_DONOTROUND)
        result2 = _DwmSetWindowAttribute(
            hwnd,
            DWMWA_WINDOW_CORNER_PREFERENCE,
            ctypes.byref(value2),
            ctypes.sizeof(value2),
        )

        return result1 == 0 and result2 == 0
    except Exception as e:
        print(f"[remove_dwm_glass_border] error: {e}")
        return False


def set_window_topmost(widget, topmost: bool = True) -> bool:
    """
    使用 Windows SetWindowPos 将窗口设为置顶或取消置顶。
    替代 Qt.WindowStaysOnTopHint，避免与 WA_TranslucentBackground 冲突。
    必须在窗口 show() 之后调用。
    """
    if sys.platform != "win32":
        print("[set_window_topmost] not on win32, skip")
        return False

    try:
        user32 = ctypes.windll.user32
        # 显式声明参数类型，确保 64 位 HWND（指针）正确处理
        user32.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND,
            wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL

        hwnd = wintypes.HWND(int(widget.winId()))
        HWND_TOPMOST = wintypes.HWND(-1)
        HWND_NOTOPMOST = wintypes.HWND(-2)
        pos_flag = HWND_TOPMOST if topmost else HWND_NOTOPMOST

        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOACTIVATE = 0x0010
        # 参考 VBnet FAQ：只用 NOMOVE|NOSIZE，并附加 NOACTIVATE，
        # 避免重新置顶时抢夺其他窗口的键盘输入焦点。
        result = user32.SetWindowPos(
            hwnd, pos_flag, 0, 0, 0, 0,
            SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE,
        )
        return result != 0
    except Exception as e:
        print(f"[set_window_topmost] error: {e}")
        return False


def set_tool_window_style(widget) -> bool:
    """
    给窗口加上 WS_EX_TOOLWINDOW 扩展样式，使其不显示在任务栏，
    同时保留普通 Window 的点击外部不关闭行为。
    必须在窗口 show() 之后调用。
    """
    if sys.platform != "win32":
        print("[set_tool_window_style] not on win32, skip")
        return False

    try:
        user32 = ctypes.windll.user32
        user32.GetWindowLongW.argtypes = [wintypes.HWND, wintypes.INT]
        user32.GetWindowLongW.restype = wintypes.LONG
        user32.SetWindowLongW.argtypes = [wintypes.HWND, wintypes.INT, wintypes.LONG]
        user32.SetWindowLongW.restype = wintypes.LONG
        user32.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND,
            wintypes.INT, wintypes.INT, wintypes.INT, wintypes.INT,
            wintypes.UINT,
        ]
        user32.SetWindowPos.restype = wintypes.BOOL

        hwnd = wintypes.HWND(int(widget.winId()))
        GWL_EXSTYLE = -20
        WS_EX_TOOLWINDOW = 0x00000080
        exstyle = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, exstyle | WS_EX_TOOLWINDOW)
        # 强制刷新非客户区，使样式立即生效；保持 SWP_NOZORDER，不破坏置顶
        SWP_NOSIZE = 0x0001
        SWP_NOMOVE = 0x0002
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        SWP_SHOWWINDOW = 0x0040
        user32.SetWindowPos(
            hwnd, wintypes.HWND(0), 0, 0, 0, 0,
            SWP_NOSIZE | SWP_NOMOVE | SWP_NOZORDER | SWP_FRAMECHANGED | SWP_SHOWWINDOW,
        )
        return True
    except Exception as e:
        print(f"[set_tool_window_style] error: {e}")
        return False
