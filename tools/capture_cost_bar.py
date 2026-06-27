import argparse
import sys
import time
from pathlib import Path

import cv2
import keyboard
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture


# 默认 ROI 比例与 main.py 中费用条检测区域一致（2560x1600 基准）
ROI_RATIOS = (
    2343 / 2560,  # x
    1278 / 1600,  # y
    (2560 - 2343) / 2560,  # w
    (1284 - 1278) / 1600,  # h
)


def capture_cost_bar_roi(cap: WindowCapture, roi_ratios=ROI_RATIOS):
    """截取费用条 ROI，返回 BGRA 图像。"""
    w, h = cap.get_window_size()
    x = int(w * roi_ratios[0])
    y = int(h * roi_ratios[1])
    rw = int(w * roi_ratios[2])
    rh = int(h * roi_ratios[3])
    left = cap.monitor.get("left", 0)
    top = cap.monitor.get("top", 0)
    return cap.capture_roi(left + x, top + y, rw, rh)


def save_frame(img, out_dir: Path, idx: int, threshold: int = 200):
    """保存截图并打印白像素统计信息。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    white_count = int(np.sum(gray > threshold))
    total = gray.size
    path = out_dir / f"frame_{idx:03d}.png"
    cv2.imencode(".png", img)[1].tofile(str(path))
    print(
        f"保存 {path.name}: 白像素(>{threshold})={white_count}/{total} "
        f"({white_count / total * 100:.1f}%)"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="费用条 ROI 截图工具")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--interval", type=float, default=33, help="自动截图间隔(ms)，默认 33")
    group.add_argument("--fps", type=float, help="自动截图帧率(fps)，与 --interval 二选一")
    parser.add_argument("--count", type=int, default=30, help="自动截图帧数，默认 30")
    parser.add_argument("--out-dir", type=Path, help="输出目录，默认 debug/cost_bar_frames")
    parser.add_argument("--threshold", type=int, default=200, help="白像素灰度阈值，默认 200")
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X_RATIO", "Y_RATIO", "W_RATIO", "H_RATIO"),
        help="自定义 ROI 比例（基于窗口宽高的比例）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cap = WindowCapture(backend="mss")

    interval_ms = args.interval if args.fps is None else (1000.0 / args.fps)
    out_dir = args.out_dir or (ROOT / "debug" / "cost_bar_frames")
    out_dir.mkdir(parents=True, exist_ok=True)
    roi_ratios = tuple(args.roi) if args.roi else ROI_RATIOS

    print(f"费用条截图将保存到: {out_dir}")
    print(f"自动截图参数: 间隔={interval_ms:.2f}ms, 帧数={args.count}, 阈值={args.threshold}")
    print("操作说明:")
    print("  F9    - 手动截取一帧")
    print(f"  P     - 自动连续截取 {args.count} 帧（按 {interval_ms:.2f}ms 间隔）")
    print("  ESC   - 退出")
    print("若 ROI 位置不对，可使用 --roi 参数或修改本文件中的 ROI_RATIOS。")

    frame_idx = 0
    while True:
        if keyboard.is_pressed("f9"):
            try:
                img = capture_cost_bar_roi(cap, roi_ratios)
                save_frame(img, out_dir, frame_idx, args.threshold)
                frame_idx += 1
            except Exception as e:
                print(f"截图失败: {e}")
            time.sleep(0.3)  # 去抖

        elif keyboard.is_pressed("p"):
            print(f"开始自动截取 {args.count} 帧，请保持游戏画面稳定...")
            for i in range(args.count):
                t0 = time.perf_counter()
                try:
                    img = capture_cost_bar_roi(cap, roi_ratios)
                    save_frame(img, out_dir, frame_idx, args.threshold)
                    frame_idx += 1
                except Exception as e:
                    print(f"第 {i + 1}/{args.count} 帧截图失败: {e}")
                elapsed_ms = (time.perf_counter() - t0) * 1000
                sleep_ms = max(0, interval_ms - elapsed_ms)
                time.sleep(sleep_ms / 1000)
            print("自动截取完成")
            time.sleep(0.5)

        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
