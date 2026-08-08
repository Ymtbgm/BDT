"""对比 mss 与 windows-capture 后端的截图性能。

默认捕获《明日方舟》窗口的倍率区域 B（ROI_B），分别用两种后端跑 3 秒，
输出每帧耗时、帧间隔等统计信息。

用法：
    python tools/benchmark_capture_backends.py
    python tools/benchmark_capture_backends.py --roi 2175 34 128 119 --duration 3
"""

import statistics
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture, _WINDOWS_CAPTURE_AVAILABLE
from core.region_state_timer import RegionStateTimer, _RateTemplateMatcher
import core.constants as constants
from core.paths import GAME_TEMPLATE_DIR

_active_cap: WindowCapture | None = None


def _cleanup(*_):
    print("\n[清理] 停止捕获...")
    global _active_cap
    if _active_cap is not None:
        _active_cap.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, _cleanup)
signal.signal(signal.SIGTERM, _cleanup)


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="截图后端性能对比")
    parser.add_argument(
        "--window-title",
        type=str,
        default="明日方舟",
        help="目标窗口标题（默认 明日方舟）",
    )
    parser.add_argument(
        "--roi",
        type=int,
        nargs=4,
        default=RegionStateTimer.DEFAULT_ROI_B,
        help="测试 ROI：x y w h（默认区域 B）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="每种后端测试时长（秒，默认 3）",
    )
    parser.add_argument(
        "--wgc-interval",
        type=int,
        default=6,
        help="windows-capture 最小更新间隔 ms（默认 6）",
    )
    return parser.parse_args()


def _fmt_stats(values_ms):
    if not values_ms:
        return "无数据"
    return (
        f"avg={statistics.mean(values_ms):.3f}ms "
        f"median={statistics.median(values_ms):.3f}ms "
        f"min={min(values_ms):.3f}ms "
        f"max={max(values_ms):.3f}ms "
        f"p95={sorted(values_ms)[int(len(values_ms) * 0.95)]:.3f}ms"
    )


def benchmark_mss(window_title: str, roi: tuple, duration_s: float):
    print("\n[后端: mss]")
    global _active_cap
    cap = WindowCapture(window_title=window_title, backend="mss")
    _active_cap = cap
    x, y, w, h = roi
    times = []

    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        img = cap.capture_roi(x, y, w, h)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
        # 控制一下空转频率，避免把 CPU 跑满导致系统调度抖动
        time.sleep(0.001)

    cap.stop()
    _active_cap = None
    print(f"  采样数: {len(times)}")
    print(f"  capture_roi 耗时: {_fmt_stats(times)}")
    print(f"  估算 FPS: {len(times) / duration_s:.1f}")
    return times


def benchmark_mss_with_rate_matching(window_title: str, roi: tuple, duration_s: float):
    print("\n[后端: mss + 区域B倍率模板匹配]")
    global _active_cap
    cap = WindowCapture(window_title=window_title, backend="mss")
    _active_cap = cap
    x, y, w, h = roi

    tmpl_dir = GAME_TEMPLATE_DIR
    matcher = _RateTemplateMatcher(
        str(tmpl_dir / constants.RATE_TEMPLATE_FAST_NAME),
        str(tmpl_dir / constants.RATE_TEMPLATE_SLOW_NAME),
        fast2x_path=str(tmpl_dir / constants.RATE_TEMPLATE_FAST2X_NAME),
    )
    if not matcher.available:
        print("  倍率模板加载失败，跳过")
        cap.stop()
        _active_cap = None
        return None

    capture_times = []
    convert_times = []
    match_times = []
    total_times = []
    states = []

    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        img = cap.capture_roi(x, y, w, h)
        t1 = time.perf_counter()

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        t2 = time.perf_counter()

        score_fast, score_slow, score_fast2x, mean_diff, state = matcher.match(gray)
        t3 = time.perf_counter()

        capture_times.append((t1 - t0) * 1000.0)
        convert_times.append((t2 - t1) * 1000.0)
        match_times.append((t3 - t2) * 1000.0)
        total_times.append((t3 - t0) * 1000.0)
        states.append(state)

        time.sleep(0.001)

    cap.stop()
    _active_cap = None

    print(f"  采样数: {len(total_times)}")
    print(f"  capture_roi 耗时: {_fmt_stats(capture_times)}")
    print(f"  BGRA->Gray 耗时:  {_fmt_stats(convert_times)}")
    print(f"  模板匹配耗时:     {_fmt_stats(match_times)}")
    print(f"  完整 pipeline:    {_fmt_stats(total_times)}")
    print(f"  估算 FPS: {len(total_times) / duration_s:.1f}")
    if states:
        print(f"  识别状态分布: fast={states.count('fast')} slow={states.count('slow')} fast2x={states.count('fast2x')} transition={states.count('transition')}")
    return total_times


def benchmark_windows_capture(window_title: str, roi: tuple, duration_s: float, interval_ms: int):
    print("\n[后端: windows-capture]")
    if not _WINDOWS_CAPTURE_AVAILABLE:
        print("  未安装 windows-capture 包，跳过")
        return None

    try:
        cap = WindowCapture(
            window_title=window_title,
            backend="windows_capture",
            minimum_update_interval=interval_ms,
        )
        _active_cap = cap
    except Exception as e:
        print(f"  启动失败: {e}")
        return None

    x, y, w, h = roi
    read_times = []

    deadline = time.perf_counter() + duration_s
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        try:
            img = cap.capture_roi(x, y, w, h)
        except Exception as e:
            print(f"  capture_roi 异常: {e}")
            break
        t1 = time.perf_counter()
        read_times.append((t1 - t0) * 1000.0)
        time.sleep(0.001)

    intervals = cap.get_frame_arrival_intervals()
    cap.stop()
    _active_cap = None
    print(f"  采样数: {len(read_times)}")
    print(f"  capture_roi 耗时: {_fmt_stats(read_times)}")
    if intervals:
        print(f"  后台捕获帧间隔: {_fmt_stats(intervals)}")
        print(f"  后台估算 FPS: {1000.0 / max(statistics.mean(intervals), 0.001):.1f}")
    print(f"  估算 FPS: {len(read_times) / duration_s:.1f}")
    return read_times


def main():
    args = _parse_args()
    roi = tuple(args.roi)
    print(f"目标窗口: {args.window_title}")
    print(f"测试 ROI: {roi}")
    print(f"测试时长: {args.duration}s")

    mss_times = benchmark_mss(args.window_title, roi, args.duration)
    mss_pipeline_times = benchmark_mss_with_rate_matching(args.window_title, roi, args.duration)
    wgc_times = benchmark_windows_capture(
        args.window_title, roi, args.duration, args.wgc_interval
    )

    print("\n[对比]")
    if mss_times:
        print(f"  mss 平均耗时: {statistics.mean(mss_times):.3f}ms")
    if mss_pipeline_times:
        print(f"  mss+模板匹配 平均耗时: {statistics.mean(mss_pipeline_times):.3f}ms")
    if wgc_times:
        print(f"  WGC 平均耗时: {statistics.mean(wgc_times):.3f}ms")
    if mss_times and wgc_times:
        speedup = statistics.mean(mss_times) / max(statistics.mean(wgc_times), 0.001)
        print(f"  WGC 读取速度约为 mss 的 {speedup:.1f} 倍")
    if mss_times and mss_pipeline_times:
        overhead = statistics.mean(mss_pipeline_times) - statistics.mean(mss_times)
        print(f"  模板匹配额外开销: {overhead:.3f}ms")


if __name__ == "__main__":
    main()
