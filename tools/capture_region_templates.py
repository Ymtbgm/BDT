import sys
import time
from pathlib import Path

import cv2
import keyboard
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture


# 默认 ROI 为 2560x1600 分辨率下的绝对屏幕坐标
# 格式: (x, y, w, h)
DEFAULT_ROIS = {
    "region_a": (2375, 53, 112, 88),
    "region_b": (2175, 34, 128, 119),
}


def capture_region(cap: WindowCapture, roi: tuple):
    """截取指定 ROI，返回 BGRA 图像。"""
    x, y, w, h = roi
    return cap.capture_roi(x, y, w, h)


def save_region(img, out_dir: Path, label: str, idx: int, threshold: int = 200):
    """保存截图并打印灰度统计信息。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    white_count = int(np.sum(gray > threshold))
    mean_val = float(np.mean(gray))
    path = out_dir / f"{label}_{idx:03d}.png"
    cv2.imencode(".png", img)[1].tofile(str(path))
    print(
        f"保存 {path.name}: 平均灰度={mean_val:.1f}, "
        f"白像素(>{threshold})={white_count}/{gray.size}"
    )


def main():
    cap = WindowCapture(backend="mss")
    out_dir = ROOT / "debug" / "region_templates"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"区域模板将保存到: {out_dir}")
    print("当前 ROI 配置:")
    for label, roi in DEFAULT_ROIS.items():
        print(f"  {label}: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print("操作说明:")
    print("  F9  - 截取两个区域当前帧并保存")
    print("  ESC - 退出")
    print("若 ROI 位置不对，可修改本文件中的 DEFAULT_ROIS。")

    frame_idx = 0
    while True:
        if keyboard.is_pressed("f9"):
            try:
                for label, roi in DEFAULT_ROIS.items():
                    img = capture_region(cap, roi)
                    save_region(img, out_dir, label, frame_idx)
                frame_idx += 1
            except Exception as e:
                print(f"截图失败: {e}")
            time.sleep(0.3)

        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
