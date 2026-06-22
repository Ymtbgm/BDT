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


def capture_cost_bar_roi(cap: WindowCapture):
    """截取费用条 ROI，返回 BGRA 图像。"""
    w, h = cap.get_window_size()
    x = int(w * ROI_RATIOS[0])
    y = int(h * ROI_RATIOS[1])
    rw = int(w * ROI_RATIOS[2])
    rh = int(h * ROI_RATIOS[3])
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


def main():
    cap = WindowCapture(backend="mss")
    out_dir = ROOT / "debug" / "cost_bar_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"费用条截图将保存到: {out_dir}")
    print("操作说明:")
    print("  F9    - 手动截取一帧")
    print("  Space - 自动连续截取 30 帧（约 1 秒，按 33ms 间隔）")
    print("  ESC   - 退出")
    print("若 ROI 位置不对，可修改本文件中的 ROI_RATIOS。")

    frame_idx = 0
    while True:
        if keyboard.is_pressed("f9"):
            try:
                img = capture_cost_bar_roi(cap)
                save_frame(img, out_dir, frame_idx)
                frame_idx += 1
            except Exception as e:
                print(f"截图失败: {e}")
            time.sleep(0.3)  # 去抖

        elif keyboard.is_pressed("space"):
            print("开始自动截取 30 帧，请保持游戏画面稳定...")
            for i in range(30):
                t0 = time.perf_counter()
                try:
                    img = capture_cost_bar_roi(cap)
                    save_frame(img, out_dir, frame_idx)
                    frame_idx += 1
                except Exception as e:
                    print(f"第 {i + 1}/30 帧截图失败: {e}")
                elapsed_ms = (time.perf_counter() - t0) * 1000
                sleep_ms = max(0, 33 - elapsed_ms)
                time.sleep(sleep_ms / 1000)
            print("自动截取完成")
            time.sleep(0.5)

        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
