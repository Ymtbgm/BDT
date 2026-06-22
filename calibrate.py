import sys
import cv2
import numpy as np

from core.capture import WindowCapture
from core.grid_mapper import GridMapper


def draw_grid(frame: np.ndarray, mapper: GridMapper):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for row in range(mapper.grid_rows):
        for col in range(mapper.grid_cols):
            cx, cy = mapper.grid_to_pixel(row, col)
            cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)
            label = f"({row},{col})"
            cv2.putText(overlay, label, (cx + 8, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

    alpha = 0.7
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)


def main():
    if len(sys.argv) < 3:
        print("用法: python calibrate.py <grid_rows> <grid_cols>")
        sys.exit(1)

    rows = int(sys.argv[1])
    cols = int(sys.argv[2])

    capture = WindowCapture()
    w, h = capture.get_window_size()
    mapper = GridMapper(w, h, rows, cols)

    frame = capture.capture()
    vis = draw_grid(frame, mapper)
    cv2.imwrite("calibration.png", vis)
    print("已保存 calibration.png，请查看图片核对格子位置。")


if __name__ == "__main__":
    main()
