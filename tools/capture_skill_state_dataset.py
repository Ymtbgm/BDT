import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.capture import WindowCapture
from core.tile_pos import TilePosCalculator


def load_stage_info(code: str | None, name: str | None) -> dict | None:
    """从 levels.json 查找关卡信息，返回包含 width/height/view 的字典。"""
    levels_path = Path(__file__).parent.parent / "core" / "resource" / "levels.json"
    if not levels_path.exists():
        return None
    try:
        levels = json.loads(levels_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    for lv in levels:
        if code and lv.get("code") == code:
            return lv
        if name and lv.get("name") == name:
            return lv
    return None


def main():
    parser = argparse.ArgumentParser(
        description="批量截取所有格子的头顶技能状态检测 ROI，用于合成训练数据集"
    )
    parser.add_argument("--stage-code", type=str, default=None, help="关卡 code")
    parser.add_argument("--stage-name", type=str, default=None, help="关卡 name")
    parser.add_argument("--rows", type=int, default=None, help="地图行数")
    parser.add_argument("--cols", type=int, default=None, help="地图列数")
    parser.add_argument("--roi-offset-x", type=int, default=-35, help="ROI 相对格子中心的 x 偏移")
    parser.add_argument("--roi-offset-y", type=int, default=-240, help="ROI 相对格子中心的 y 偏移")
    parser.add_argument("--roi-w", type=int, default=75, help="ROI 宽度")
    parser.add_argument("--roi-h", type=int, default=75, help="ROI 高度")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出文件夹路径（默认 debug/skill_state_dataset/<timestamp>）",
    )
    parser.add_argument("--prefix", type=str, default="roi", help="文件名前缀")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="同时保存一张带 ROI 矩形的全屏叠加图",
    )
    args = parser.parse_args()

    # 解析关卡尺寸
    stage_info = None
    if args.stage_code or args.stage_name:
        stage_info = load_stage_info(args.stage_code, args.stage_name)
        if stage_info is None:
            print(f"警告：未在 levels.json 中找到关卡 code={args.stage_code} name={args.stage_name}")

    if stage_info is not None:
        if args.rows is None:
            args.rows = stage_info.get("height")
        if args.cols is None:
            args.cols = stage_info.get("width")

    rows = args.rows or 7
    cols = args.cols or 9
    print(f"使用地图尺寸: {rows}x{cols}")

    capture = WindowCapture()
    full = capture.capture()
    h, w = full.shape[:2]
    print(f"窗口尺寸: {w}x{h}")

    scale = min(w / 2560, h / 1600)
    ox = int(args.roi_offset_x * scale)
    oy = int(args.roi_offset_y * scale)
    roi_w = max(1, int(args.roi_w * scale))
    roi_h = max(1, int(args.roi_h * scale))
    print(f"ROI 缩放系数: {scale:.3f}，实际偏移: ({ox},{oy})，实际尺寸: {roi_w}x{roi_h}")

    tile_calc = TilePosCalculator(
        screen_width=w,
        screen_height=h,
        grid_rows=rows,
        grid_cols=cols,
        stage_code=args.stage_code,
        stage_name=args.stage_name,
    )

    if args.output:
        out_dir = Path(args.output)
    else:
        out_dir = Path("debug") / "skill_state_dataset" / str(int(time.time() * 1000))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir.absolute()}")

    left = capture.monitor.get("left", 0)
    top = capture.monitor.get("top", 0)

    canvas = full.copy()
    if canvas.shape[2] == 4:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGRA2BGR)

    count = 0
    for r in range(rows):
        for c in range(cols):
            center_x, center_y = tile_calc.get_screen_pos(r, c, side=False)
            abs_x = left + center_x + ox
            abs_y = top + center_y + oy

            try:
                roi = capture.capture_roi(abs_x, abs_y, roi_w, roi_h)
            except Exception as e:
                print(f"跳过 ({r},{c}): 截取失败 {e}")
                continue

            ts = int(time.time() * 1000000)
            filename = f"{args.prefix}_{r}_{c}_{ts}.png"
            path = out_dir / filename
            try:
                ok, encoded = cv2.imencode(".png", roi)
                if ok:
                    path.write_bytes(encoded.tobytes())
                    count += 1
            except Exception as e:
                print(f"保存失败 {path}: {e}")
                continue

            if args.visualize:
                rx = center_x + ox
                ry = center_y + oy
                cv2.rectangle(canvas, (rx, ry), (rx + roi_w, ry + roi_h), (0, 255, 255), 1)
                cv2.putText(
                    canvas,
                    f"{r},{c}",
                    (rx, max(0, ry - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 255),
                    1,
                )

            print(f"已保存 ({r},{c}) -> {filename}")

    if args.visualize:
        vis_path = out_dir / "visualization.png"
        try:
            ok, encoded = cv2.imencode(".png", canvas)
            if ok:
                vis_path.write_bytes(encoded.tobytes())
                print(f"已保存可视化图: {vis_path}")
        except Exception as e:
            print(f"可视化图保存失败: {e}")

    print(f"完成，共保存 {count} 张 ROI")


if __name__ == "__main__":
    main()
