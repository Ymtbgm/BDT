"""Windows Graphics Capture 动态内容基准测试。

通过 tkinter 生成高速变化的测试画面，对比 WGC 后台实际捕获帧率与画面
更新频率是否匹配。支持两种模式：

- moving_rect：移动彩色方块（原有测试）
- random_dots：黑底随机白点，每帧都变化，用于验证 165Hz 下的 6ms 间隔

用法：
    python tools/windows_capture_dynamic_benchmark.py
    python tools/windows_capture_dynamic_benchmark.py --mode random_dots --update-interval 6
"""

import statistics
import sys
import threading
import time
import tkinter as tk
from pathlib import Path

import numpy as np
import win32gui
import win32con

# 请求 1ms 系统定时器分辨率，尽量让 tkinter 的 after(6) 更接近 6ms
try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl, CaptureControl
except ImportError as e:
    print(f"[错误] 无法导入 windows_capture: {e}")
    print("请确认已在当前环境安装: pip install windows-capture")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REGION = (100, 100, 228, 219)  # left, top, right, bottom
DURATION = 5.0


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="WGC 动态内容基准测试")
    parser.add_argument(
        "--mode",
        type=str,
        default="random_dots",
        choices=["moving_rect", "random_dots"],
        help="测试画面模式（默认 random_dots）",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=6,
        help="tkinter 画面更新间隔 ms（默认 6，对应 165Hz）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="测试时长（秒，默认 5）",
    )
    parser.add_argument(
        "--monitor-index",
        type=int,
        default=1,
        help="捕获的显示器索引（默认 1，与 --use-window 互斥）",
    )
    parser.add_argument(
        "--use-window",
        action="store_true",
        help="按 tkinter 窗口句柄捕获，而非整屏",
    )
    parser.add_argument(
        "--region",
        type=int,
        nargs=4,
        default=REGION,
        help="捕获 ROI：left top right bottom",
    )
    return parser.parse_args()


def start_benchmark(
    done_event: threading.Event,
    result_container: dict,
    duration: float,
    region: tuple,
    tk_update_count: list,
    monitor_index: int = 1,
    hwnd: int | None = None,
):
    lock = threading.Lock()
    latest_frame = None
    frame_times = []
    read_times = []
    frame_max_values = []
    frame_mean_values = []
    frame_diff_sums = []

    if hwnd is not None:
        print(f"[WGC] 按窗口句柄捕获: {hwnd}")
        capture = WindowsCapture(
            window_hwnd=hwnd,
            minimum_update_interval=1,
            cursor_capture=False,
            draw_border=False,
        )
    else:
        print(f"[WGC] 按显示器索引捕获: {monitor_index}")
        capture = WindowsCapture(
            monitor_index=monitor_index,
            minimum_update_interval=1,
            cursor_capture=False,
            draw_border=False,
        )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        arr = np.array(frame.frame_buffer)
        with lock:
            nonlocal latest_frame
            latest_frame = arr
            frame_times.append(time.perf_counter())
            frame_max_values.append(int(arr.max()))
            frame_mean_values.append(float(arr.mean()))

    @capture.event
    def on_closed():
        pass

    control: CaptureControl = capture.start_free_threaded()

    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        with lock:
            arr = latest_frame
        if arr is not None:
            cropped = arr[region[1]:region[3], region[0]:region[2]]
            _ = cropped.shape
        t1 = time.perf_counter()
        read_times.append((t1 - t0) * 1000.0)
        time.sleep(0.001)

    control.stop()
    control.wait()

    result_container["frame_times"] = frame_times
    result_container["read_times"] = read_times
    result_container["frame_max_values"] = frame_max_values
    result_container["frame_mean_values"] = frame_mean_values
    done_event.set()


def run_moving_rect(root: tk.Tk, done_event: threading.Event, update_interval: int, tk_update_count: list):
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.configure(bg="black")

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
    canvas.pack()

    rect = canvas.create_rectangle(0, 0, 200, 200, fill="red", outline="")

    colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "white", "black"]
    x, y = 0, 0
    dx, dy = 30, 25
    color_idx = 0

    def update():
        nonlocal x, y, dx, dy, color_idx
        tk_update_count[0] += 1
        if done_event.is_set():
            root.destroy()
            return
        x += dx
        y += dy
        if x < 0 or x > screen_w - 200:
            dx = -dx
            x += dx
        if y < 0 or y > screen_h - 200:
            dy = -dy
            y += dy
        color_idx = (color_idx + 1) % len(colors)
        canvas.coords(rect, x, y, x + 200, y + 200)
        canvas.itemconfig(rect, fill=colors[color_idx])
        root.after(update_interval, update)

    root.after(update_interval, update)


def run_random_dots(root: tk.Tk, done_event: threading.Event, update_interval: int, tk_update_count: list):
    """黑底随机白点：每帧清除旧点并画新点，确保 WGC 每帧都有变化。"""
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.geometry(f"{screen_w}x{screen_h}+0+0")
    root.configure(bg="black")

    canvas = tk.Canvas(root, width=screen_w, height=screen_h, bg="black", highlightthickness=0)
    canvas.pack()

    dot_ids = []

    def update():
        if done_event.is_set():
            root.destroy()
            return

        tk_update_count[0] += 1

        # 清除上一帧的白点
        for dot_id in dot_ids:
            canvas.delete(dot_id)
        dot_ids.clear()

        # 随机画一个 8x8 白点；也可以多画几个增加变化面积
        x = int(np.random.randint(0, screen_w - 8))
        y = int(np.random.randint(0, screen_h - 8))
        dot_id = canvas.create_rectangle(x, y, x + 8, y + 8, fill="white", outline="white")
        dot_ids.append(dot_id)

        root.after(update_interval, update)

    root.after(update_interval, update)


def main():
    args = _parse_args()

    root = tk.Tk()
    root.title("WGC Dynamic Benchmark")
    root.configure(bg="black")
    # 让窗口真正创建并显示，获取有效 HWND
    root.geometry("800x600+0+0")
    root.deiconify()
    root.lift()
    root.attributes("-topmost", True)
    root.update()
    raw_hwnd = root.winfo_id()
    # WGC 需要顶层窗口句柄
    hwnd = win32gui.GetAncestor(raw_hwnd, win32con.GA_ROOT) if args.use_window else None
    if args.use_window:
        print(f"[WGC] raw HWND={raw_hwnd}, root HWND={hwnd}")

    done_event = threading.Event()
    result_container = {}
    tk_update_count = [0]
    bench_thread = threading.Thread(
        target=start_benchmark,
        args=(done_event, result_container, args.duration, tuple(args.region), tk_update_count),
        kwargs={"monitor_index": args.monitor_index, "hwnd": hwnd},
        daemon=True,
    )
    bench_thread.start()

    if args.mode == "moving_rect":
        run_moving_rect(root, done_event, args.update_interval, tk_update_count)
    else:
        run_random_dots(root, done_event, args.update_interval, tk_update_count)

    root.mainloop()
    bench_thread.join()

    frame_times = result_container.get("frame_times", [])
    read_times = result_container.get("read_times", [])
    frame_max_values = result_container.get("frame_max_values", [])
    frame_mean_values = result_container.get("frame_mean_values", [])

    capture_mode = "窗口句柄" if args.use_window else f"显示器 {args.monitor_index}"
    print(f"\n模式: {args.mode}, 画面更新间隔: {args.update_interval}ms, 测试时长: {args.duration}s, 捕获方式: {capture_mode}")
    print(f"tkinter 实际 update 调用次数: {tk_update_count[0]}")
    print(f"tkinter 实际更新 FPS: {tk_update_count[0] / args.duration:.1f}")

    if frame_max_values:
        # 用平均像素值变化作为“内容是否变化”的廉价判断
        mean_changes = sum(
            1 for i in range(1, len(frame_mean_values))
            if abs(frame_mean_values[i] - frame_mean_values[i - 1]) > 0.1
        )
        print(f"\nWGC 帧内容分析:")
        print(f"  总帧数: {len(frame_max_values)}")
        print(f"  与上一帧平均亮度有差异的帧数: {mean_changes} ({mean_changes / max(1, len(frame_max_values) - 1) * 100:.1f}%)")
        print(f"  最大像素值: min={min(frame_max_values)} max={max(frame_max_values)} avg={statistics.mean(frame_max_values):.1f}")
        print(f"  平均像素值: min={min(frame_mean_values):.1f} max={max(frame_mean_values):.1f} avg={statistics.mean(frame_mean_values):.1f}")

    if frame_times and len(frame_times) > 1:
        intervals = [
            (frame_times[i] - frame_times[i - 1]) * 1000.0
            for i in range(1, len(frame_times))
        ]
        print(f"\nWGC 后台捕获帧数: {len(frame_times)}")
        print(
            f"帧间隔 (ms): "
            f"avg={statistics.mean(intervals):.3f} "
            f"median={statistics.median(intervals):.3f} "
            f"min={min(intervals):.3f} "
            f"max={max(intervals):.3f} "
            f"p95={sorted(intervals)[int(len(intervals) * 0.95)]:.3f}"
        )
        print(f"后台估算 FPS: {1000.0 / max(statistics.mean(intervals), 0.001):.1f}")
    else:
        print("\nWGC 未捕获到帧")

    if read_times:
        print(
            f"\n缓存读取+crop 耗时 ({len(read_times)} 次迭代)\n"
            f"avg={statistics.mean(read_times):.3f}ms "
            f"median={statistics.median(read_times):.3f}ms "
            f"min={min(read_times):.3f}ms "
            f"max={max(read_times):.3f}ms "
            f"p95={sorted(read_times)[int(len(read_times) * 0.95)]:.3f}ms"
        )


if __name__ == "__main__":
    main()
