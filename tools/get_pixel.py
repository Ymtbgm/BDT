import tkinter as tk
from tkinter import ttk
import ctypes
import win32gui
import win32api
import keyboard

# 开启 DPI Awareness，避免 Windows 缩放导致坐标/尺寸偏差
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()


def get_window_info(title="明日方舟"):
    hwnd = win32gui.FindWindow(None, title)
    if hwnd == 0:
        hwnd = win32gui.FindWindow(None, "Arknights")
    if hwnd == 0:
        return None

    # 完整窗口大小（含边框、标题栏）
    wl, wt, wr, wb = win32gui.GetWindowRect(hwnd)
    win_w, win_h = wr - wl, wb - wt

    # 客户区大小（实际游戏渲染区域）
    cl, ct, cr, cb = win32gui.GetClientRect(hwnd)
    cl, ct = win32gui.ClientToScreen(hwnd, (cl, ct))
    cr, cb = win32gui.ClientToScreen(hwnd, (cr, cb))
    client_w, client_h = cr - cl, cb - ct

    return {
        "win_left": wl, "win_top": wt,
        "win_w": win_w, "win_h": win_h,
        "client_left": cl, "client_top": ct,
        "client_w": client_w, "client_h": client_h,
    }


class PixelTool:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("像素坐标获取工具")
        self.root.geometry("420x440")
        self.root.attributes("-topmost", True)
        self.root.resizable(False, False)

        self.records = []
        self._build_ui()
        self._start_polling()
        self._bind_hotkeys()

    def _build_ui(self):
        frame = ttk.Frame(self.root, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="实时坐标", font=("Microsoft YaHei", 12, "bold")).pack(anchor=tk.W)

        self.lbl_global = ttk.Label(frame, text="屏幕绝对: (0, 0)", font=("Consolas", 11))
        self.lbl_global.pack(anchor=tk.W, pady=2)

        self.lbl_relative = ttk.Label(frame, text="客户区相对: (0, 0)", font=("Consolas", 11))
        self.lbl_relative.pack(anchor=tk.W, pady=2)

        self.lbl_window = ttk.Label(frame, text="窗口: 未找到", font=("Consolas", 10), foreground="gray")
        self.lbl_window.pack(anchor=tk.W, pady=2)

        self.lbl_screen = ttk.Label(frame, text="屏幕分辨率: -", font=("Consolas", 9), foreground="gray")
        self.lbl_screen.pack(anchor=tk.W, pady=2)

        ttk.Separator(frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=8)

        ttk.Label(frame, text="记录列表 (按 F9 记录)", font=("Microsoft YaHei", 12, "bold")).pack(anchor=tk.W)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)

        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.listbox = tk.Listbox(list_frame, font=("Consolas", 10), yscrollcommand=scrollbar.set)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=4)
        ttk.Button(btn_frame, text="复制选中", command=self._copy_selected).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="清空记录", command=self._clear_records).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="复制全部", command=self._copy_all).pack(side=tk.LEFT, padx=2)

        ttk.Label(frame, text="提示: F9=记录坐标  F10=复制最新  Esc=退出", font=("Microsoft YaHei", 9), foreground="gray").pack(anchor=tk.W, pady=4)

    def _bind_hotkeys(self):
        keyboard.add_hotkey("f9", self._record_current)
        keyboard.add_hotkey("f10", self._copy_latest)
        keyboard.add_hotkey("esc", self._quit)

    def _quit(self):
        self.root.destroy()

    def _start_polling(self):
        self._update_coords()

    def _update_coords(self):
        x, y = win32api.GetCursorPos()
        info = get_window_info()

        self.lbl_global.config(text=f"屏幕绝对: ({x}, {y})")

        sw = win32api.GetSystemMetrics(0)
        sh = win32api.GetSystemMetrics(1)
        self.lbl_screen.config(text=f"屏幕分辨率: {sw} x {sh}")

        if info:
            rel_x, rel_y = x - info["client_left"], y - info["client_top"]
            self.lbl_relative.config(
                text=f"客户区相对: ({rel_x}, {rel_y})"
            )
            self.lbl_window.config(
                text=f"窗口: {info['win_w']}x{info['win_h']} (含边框) | 客户区: {info['client_w']}x{info['client_h']}",
                foreground="green"
            )
        else:
            self.lbl_relative.config(text="客户区相对: (N/A)")
            self.lbl_window.config(text="窗口: 未找到", foreground="red")

        self.root.after(50, self._update_coords)

    def _record_current(self):
        x, y = win32api.GetCursorPos()
        info = get_window_info()
        if info:
            rel_x, rel_y = x - info["client_left"], y - info["client_top"]
            entry = f"绝对({x}, {y})  客户区相对({rel_x}, {rel_y})"
        else:
            entry = f"绝对({x}, {y})"
        self.records.append(entry)
        self.listbox.insert(tk.END, entry)
        self.listbox.see(tk.END)

    def _copy_selected(self):
        selection = self.listbox.curselection()
        if selection:
            text = self.listbox.get(selection[0])
            self.root.clipboard_clear()
            self.root.clipboard_append(text)

    def _copy_latest(self):
        if not self.records:
            return
        text = self.records[-1]
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _copy_all(self):
        if not self.records:
            return
        text = "\n".join(self.records)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)

    def _clear_records(self):
        self.records.clear()
        self.listbox.delete(0, tk.END)

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = PixelTool()
    app.run()
