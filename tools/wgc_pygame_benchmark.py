"""Pygame 高刷新率窗口 + WGC 捕获验证。

用 Pygame 生成一个以 165 FPS 翻转的窗口，每帧内容都变化，
然后用 WGC 按窗口句柄捕获，验证 WGC 是否能跟上 Pygame 的呈现帧率。

用法：
    python tools/wgc_pygame_benchmark.py
    python tools/wgc_pygame_benchmark.py --duration 5 --fps 165
"""

import statistics
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pygame

# 请求 1ms 系统定时器分辨率
try:
    import ctypes
    ctypes.windll.winmm.timeBeginPeriod(1)
except Exception:
    pass

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl, CaptureControl
    _WINDOWS_CAPTURE_AVAILABLE = True
except ImportError as e:
    print(f"[错误] 无法导入 windows_capture: {e}")
    sys.exit(1)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Pygame + WGC 高刷新率验证")
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="测试时长（秒，默认 5）",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=165,
        help="Pygame 目标刷新率（默认 165）",
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        default=[800, 600],
        help="窗口大小 width height（默认 800 600）",
    )
    parser.add_argument(
        "--wgc-interval",
        type=int,
        default=6,
        help="WGC minimum_update_interval ms（默认 1）",
    )
    return parser.parse_args()


def start_capture(
    done_event: threading.Event,
    result_container: dict,
    duration: float,
    hwnd: int,
    wgc_interval: int,
):
    lock = threading.Lock()
    frame_times = []
    frame_max_values = []
    frame_mean_values = []
    start_time = time.perf_counter()
    stopped = False

    capture = WindowsCapture(
        window_hwnd=hwnd,
        minimum_update_interval=wgc_interval,
        cursor_capture=False,
        draw_border=False,
    )

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        nonlocal stopped
        arr = np.array(frame.frame_buffer)
        with lock:
            frame_times.append(time.perf_counter())
            frame_max_values.append(int(arr.max()))
            frame_mean_values.append(float(arr.mean()))
        if not stopped and time.perf_counter() - start_time > duration:
            stopped = True
            capture_control.stop()

    @capture.event
    def on_closed():
        pass

    capture.start()

    result_container["frame_times"] = frame_times
    result_container["frame_max_values"] = frame_max_values
    result_container["frame_mean_values"] = frame_mean_values
    done_event.set()


def main():
    args = _parse_args()

    pygame.init()
    screen = pygame.display.set_mode(tuple(args.size), pygame.DOUBLEBUF)
    pygame.display.set_caption("WGC Pygame Benchmark")
    clock = pygame.time.Clock()

    # 获取 Windows HWND
    wm_info = pygame.display.get_wm_info()
    hwnd = wm_info.get("window")
    if hwnd is None:
        print("[错误] 无法获取 Pygame 窗口 HWND")
        pygame.quit()
        return
    print(f"[Pygame] 窗口 HWND: {hwnd}")

    done_event = threading.Event()
    result_container = {}
    capture_thread = threading.Thread(
        target=start_capture,
        args=(done_event, result_container, args.duration, hwnd, args.wgc_interval),
        daemon=True,
    )
    capture_thread.start()

    frame_count = 0
    start = time.perf_counter()
    deadline = start + args.duration

    while time.perf_counter() < deadline:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                break

        screen.fill((0, 0, 0))
        x = np.random.randint(0, args.size[0] - 32)
        y = np.random.randint(0, args.size[1] - 32)
        pygame.draw.rect(screen, (255, 255, 255), (x, y, 32, 32))
        pygame.display.flip()
        frame_count += 1
        clock.tick(args.fps)

    elapsed = time.perf_counter() - start
    done_event.set()
    capture_thread.join()

    pygame.quit()

    frame_times = result_container.get("frame_times", [])
    frame_max_values = result_container.get("frame_max_values", [])
    frame_mean_values = result_container.get("frame_mean_values", [])

    print(f"\nPygame 目标 FPS: {args.fps}, WGC minimum_update_interval: {args.wgc_interval}ms, 测试时长: {args.duration}s")
    print(f"Pygame 实际 flip 次数: {frame_count}")
    print(f"Pygame 实际 FPS: {frame_count / elapsed:.1f}")

    if frame_max_values:
        print(f"\nWGC 帧内容分析:")
        print(f"  总帧数: {len(frame_max_values)}")
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


if __name__ == "__main__":
    main()
