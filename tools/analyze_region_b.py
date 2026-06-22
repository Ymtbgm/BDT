import sys
import time
from pathlib import Path
from statistics import mean, median

import cv2
import keyboard
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture


# 区域 B 默认 ROI (2560x1600 绝对屏幕坐标)
ROI_B = (2175, 34, 128, 119)
THRESHOLD = 200


def white_count(img: np.ndarray, threshold: int = THRESHOLD) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    return int(np.sum(gray > threshold))


def capture_30_frames(cap: WindowCapture, roi: tuple):
    """在约 1 秒内连续截取 30 帧区域 B，返回 (图像列表, 白像素列表, 耗时ms)。"""
    images = []
    counts = []
    start = time.perf_counter()
    for i in range(30):
        t0 = time.perf_counter()
        img = cap.capture_roi(*roi)
        images.append(img)
        counts.append(white_count(img))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        sleep_ms = max(0, 1000 / 30 - elapsed_ms)
        time.sleep(sleep_ms / 1000)
    total_ms = (time.perf_counter() - start) * 1000
    return images, counts, total_ms


def analyze(counts: list, slow_threshold: int = 1200):
    print("\n========== 区域 B 白像素分析 ==========")
    print(f"总帧数: {len(counts)}")
    print(f"最小值: {min(counts)}")
    print(f"最大值: {max(counts)}")
    print(f"平均值: {mean(counts):.1f}")
    print(f"中位数: {median(counts):.1f}")
    below = [(i, c) for i, c in enumerate(counts) if c < slow_threshold]
    print(f"低于 {slow_threshold} 的帧数: {len(below)}")
    if below:
        print("低于阈值的帧序号:")
        for idx, c in below:
            print(f"  frame {idx:02d}: {c}")
    print("=======================================\n")


def main():
    cap = WindowCapture(backend="mss")
    out_dir = ROOT / "debug" / "region_b_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"区域 B 连续截图分析工具")
    print(f"ROI: {ROI_B}")
    print(f"截图保存到: {out_dir}")
    print("操作说明:")
    print("  F9  - 连续截取 30 帧（约 1 秒）并分析白像素")
    print("  ESC - 退出")

    batch = 0
    while True:
        if keyboard.is_pressed("f9"):
            print(f"\n开始第 {batch + 1} 组 30 帧截图...")
            images, counts, total_ms = capture_30_frames(cap, ROI_B)
            print(f"实际耗时: {total_ms:.1f}ms")
            analyze(counts)

            for i, img in enumerate(images):
                path = out_dir / f"batch_{batch:03d}_frame_{i:02d}.png"
                cv2.imencode(".png", img)[1].tofile(str(path))
            print(f"已保存到: {out_dir / f'batch_{batch:03d}_frame_*.png'}\n")

            batch += 1
            time.sleep(0.5)

        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
