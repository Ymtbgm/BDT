import sys
import cv2
import numpy as np

from core.capture import WindowCapture
from core.tile_pos import TilePosCalculator


def main():
    import argparse
    parser = argparse.ArgumentParser(description="格子坐标对齐检查")
    parser.add_argument("rows", type=int, help="地图行数")
    parser.add_argument("cols", type=int, help="地图列数")
    parser.add_argument("--side", type=int, default=0, help="0=普通视角, 1=放置视角")
    parser.add_argument("--code", type=str, default=None, help="关卡代号，如 1-7")
    parser.add_argument("--name", type=str, default=None, help="关卡名称")
    args = parser.parse_args()

    rows, cols = args.rows, args.cols
    side = bool(args.side)

    capture = WindowCapture()
    w, h = capture.get_window_size()
    print(f"窗口客户区尺寸: {w}x{h}")

    calc = TilePosCalculator(w, h, rows, cols, stage_code=args.code, stage_name=args.name)
    print(f"使用 view_normal: {calc.view_normal}")
    print(f"使用 view_side:   {calc.view_side}")

    frame = capture.capture()
    overlay = frame.copy()

    # 标出所有格子的中心点与坐标
    for r in range(rows):
        for c in range(cols):
            x, y = calc.get_screen_pos(r, c, side=side)
            # 小圆点
            cv2.circle(overlay, (x, y), 4, (0, 255, 255), -1)
            # 坐标标签，偶数列放右侧，奇数列放左侧，减少重叠
            label = f"({r},{c})"
            tx = x + 6 if c % 2 == 0 else x - 50
            ty = y - 6 if r % 2 == 0 else y + 15
            cv2.putText(overlay, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            print(f"  ({r},{c}) -> 屏幕 ({x}, {y})")

    alpha = 0.6
    vis = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imwrite("alignment_check.png", vis)
    print("已保存 alignment_check.png，请查看图片核对格子位置。")
    print(f"视角: {'放置干员(side=True)' if side else '普通视角(side=False)'}")


if __name__ == "__main__":
    main()
