import statistics
import threading
import time
import tkinter as tk

import numpy as np
from windows_capture import WindowsCapture, Frame, InternalCaptureControl, CaptureControl

REGION = (100, 100, 228, 219)  # left, top, right, bottom
DURATION = 5.0


def start_benchmark(hwnd: int, done_event: threading.Event, result_container: dict):
    lock = threading.Lock()
    latest_frame = None
    frame_times = []
    read_times = []

    capture = WindowsCapture(window_hwnd=hwnd, minimum_update_interval=0)

    @capture.event
    def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
        arr = np.array(frame.frame_buffer)
        with lock:
            nonlocal latest_frame
            latest_frame = arr
            frame_times.append(time.perf_counter())

    @capture.event
    def on_closed():
        pass

    control: CaptureControl = capture.start_free_threaded()

    deadline = time.perf_counter() + DURATION
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        with lock:
            arr = latest_frame
        if arr is not None:
            # Crop relative to window client area
            cropped = arr[REGION[1]:REGION[3], REGION[0]:REGION[2]]
            _ = cropped.shape
        t1 = time.perf_counter()
        read_times.append((t1 - t0) * 1000.0)
        time.sleep(0.004)

    control.stop()
    control.wait()

    result_container["frame_times"] = frame_times
    result_container["read_times"] = read_times
    done_event.set()


def main():
    root = tk.Tk()
    root.title("Dynamic Window for WGC Benchmark")
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

    # Get HWND after window is realized
    root.update_idletasks()
    hwnd = root.winfo_id()
    print(f"Capturing window HWND={hwnd}")

    done_event = threading.Event()
    result_container = {}
    bench_thread = threading.Thread(
        target=start_benchmark, args=(hwnd, done_event, result_container)
    )
    bench_thread.start()

    def update():
        nonlocal x, y, dx, dy, color_idx
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
        root.after(8, update)

    root.after(8, update)
    root.mainloop()
    bench_thread.join()

    frame_times = result_container.get("frame_times", [])
    read_times = result_container.get("read_times", [])

    if frame_times and len(frame_times) > 1:
        intervals = [
            (frame_times[i] - frame_times[i - 1]) * 1000.0
            for i in range(1, len(frame_times))
        ]
        print(f"\nBackend frames: {len(frame_times)}")
        print(
            f"Frame intervals (ms): "
            f"avg={statistics.mean(intervals):.3f} "
            f"median={statistics.median(intervals):.3f} "
            f"min={min(intervals):.3f} "
            f"max={max(intervals):.3f} "
            f"p95={sorted(intervals)[int(len(intervals) * 0.95)]:.3f}"
        )
    else:
        print("No backend frames captured")

    if read_times:
        print(
            f"\nCached read+crop ({len(read_times)} iterations, ~4ms polling)\n"
            f"avg={statistics.mean(read_times):.3f}ms "
            f"median={statistics.median(read_times):.3f}ms "
            f"min={min(read_times):.3f}ms "
            f"max={max(read_times):.3f}ms "
            f"p95={sorted(read_times)[int(len(read_times) * 0.95)]:.3f}ms"
        )


if __name__ == "__main__":
    main()
