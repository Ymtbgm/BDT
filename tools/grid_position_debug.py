import argparse
import json
import sys
import time
from pathlib import Path

# 让脚本在 tools/ 目录下也能找到 core 包
sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from core.capture.capture import WindowCapture
from core.map.tile_pos import TilePosCalculator


def draw_grid_positions(
    image: np.ndarray,
    tile_calc: TilePosCalculator,
    side: bool = False,
    color: tuple = (0, 0, 255),
    label_color: tuple = (0, 255, 0),
) -> np.ndarray:
    """在截图上绘制每个格子计算出的屏幕位置。"""
    canvas = image.copy()
    rows = tile_calc.grid_rows
    cols = tile_calc.grid_cols

    for r in range(rows):
        for c in range(cols):
            x, y = tile_calc.get_screen_pos(r, c, side=side)
            # 绘制十字准星
            cv2.drawMarker(
                canvas,
                (x, y),
                color,
                markerType=cv2.MARKER_CROSS,
                markerSize=20,
                thickness=2,
            )
            # 绘制小圆点
            cv2.circle(canvas, (x, y), 3, color, -1)
            # 标注行列号
            label = f"{r},{c}"
            cv2.putText(
                canvas,
                label,
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                label_color,
                1,
            )

    # 顶部信息栏
    info = f"side={side}  grid={rows}x{cols}  view_normal={tile_calc.view_normal}  view_side={tile_calc.view_side}"
    cv2.putText(
        canvas,
        info,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return canvas


def draw_tile_polygons(
    image: np.ndarray,
    tile_calc: TilePosCalculator,
    side: bool = False,
    outline_color: tuple = (0, 255, 255),
    fill_color: tuple = (0, 255, 255),
    thickness: int = 2,
    alpha: float = 0.08,
) -> np.ndarray:
    """在截图上绘制每个格子的投影四边形。"""
    canvas = image.copy().astype(np.float32)
    rows = tile_calc.grid_rows
    cols = tile_calc.grid_cols

    # 1) 半透明填充（避免盖住底层画面）
    overlay = canvas.copy()
    for r in range(rows):
        for c in range(cols):
            poly = np.array(tile_calc.get_tile_polygon(r, c, side=side), dtype=np.int32)
            poly = poly.reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [poly], fill_color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # 2) 不透明边框（确保格子边界清晰可见）
    canvas = canvas.astype(np.uint8)
    for r in range(rows):
        for c in range(cols):
            poly = np.array(tile_calc.get_tile_polygon(r, c, side=side), dtype=np.int32)
            poly = poly.reshape((-1, 1, 2))
            cv2.polylines(canvas, [poly], True, outline_color, thickness)
    return canvas


def draw_deploy_roi(
    image: np.ndarray,
    deploy_x: int,
    deploy_y: int,
    offset_x: int,
    offset_y: int,
    roi_w: int,
    roi_h: int,
) -> np.ndarray:
    """在截图上绘制部署位置及对应弹药图标 ROI 区域。"""
    canvas = image.copy()
    h, w = canvas.shape[:2]

    # 部署位置十字
    cv2.drawMarker(
        canvas,
        (deploy_x, deploy_y),
        (0, 255, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=24,
        thickness=2,
    )
    cv2.circle(canvas, (deploy_x, deploy_y), 4, (0, 255, 0), -1)
    cv2.putText(
        canvas,
        f"deploy ({deploy_x},{deploy_y})",
        (deploy_x + 10, deploy_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    x1 = deploy_x + offset_x
    y1 = deploy_y + offset_y
    x2 = min(w, x1 + roi_w)
    y2 = min(h, y1 + roi_h)
    x1 = max(0, x1)
    y1 = max(0, y1)

    cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.putText(
        canvas,
        f"ROI ({offset_x},{offset_y}) {roi_w}x{roi_h}",
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
    )
    print(f"部署位置: ({deploy_x},{deploy_y})")
    print(f"弹药 ROI: ({x1},{y1},{x2 - x1},{y2 - y1})  偏移: ({offset_x},{offset_y})")
    return canvas
from core.base.paths import game_data

def load_stage_info(code: str | None, name: str | None) -> dict | None:
    """从 levels.json 查找关卡信息，返回包含 width/height/view 的字典。"""
    levels_path = game_data("levels.json")
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
    parser = argparse.ArgumentParser(description="绘制并保存 side=False 下各格子的计算位置")
    parser.add_argument("--rows", type=int, default=None, help="地图行数（默认从 levels.json 读取）")
    parser.add_argument("--cols", type=int, default=None, help="地图列数（默认从 levels.json 读取）")
    parser.add_argument("--stage-code", type=str, default=None, help="关卡 code，用于从 levels.json 加载精确 view 和尺寸")
    parser.add_argument("--stage-name", type=str, default=None, help="关卡 name，用于从 levels.json 加载精确 view 和尺寸")
    parser.add_argument("--side", action="store_true", help="同时绘制 side=True 的位置（蓝色）")
    parser.add_argument("--polygons", action="store_true", help="同时绘制每个格子的投影四边形（半透明填充+边框）")
    parser.add_argument("--output", type=str, default=None, help="输出图片路径")
    parser.add_argument("--deploy-row", type=int, default=None, help="部署位置所在行（格子坐标）")
    parser.add_argument("--deploy-col", type=int, default=None, help="部署位置所在列（格子坐标）")
    parser.add_argument("--roi-offset-x", type=int, default=-35, help="ROI 相对于部署位置的 x 偏移（默认 -20）")
    parser.add_argument("--roi-offset-y", type=int, default=-240, help="ROI 相对于部署位置的 y 偏移（默认 -240）")
    parser.add_argument("--roi-w", type=int, default=75, help="ROI 宽度（默认 75）")
    parser.add_argument("--roi-h", type=int, default=75, help="ROI 高度（默认 75）")
    args = parser.parse_args()

    # 尝试从 levels.json 解析关卡尺寸
    stage_info = None
    if args.stage_code or args.stage_name:
        stage_info = load_stage_info(args.stage_code, args.stage_name)
        if stage_info is None:
            print(f"警告：未在 levels.json 中找到关卡 code={args.stage_code} name={args.stage_name}，使用默认/手动尺寸")

    if stage_info is not None:
        json_rows = stage_info.get("height")
        json_cols = stage_info.get("width")
        if args.rows is None and isinstance(json_rows, int):
            args.rows = json_rows
        if args.cols is None and isinstance(json_cols, int):
            args.cols = json_cols
        if args.rows is None or args.cols is None:
            print(f"警告：levels.json 中该关卡尺寸不完整（height={json_rows}, width={json_cols}），使用默认尺寸")

    # 仍未指定则使用默认 7x9
    rows = args.rows or 7
    cols = args.cols or 9

    capture = WindowCapture()
    image = capture.capture()
    h, w = image.shape[:2]
    print(f"窗口尺寸: {w}x{h}")

    # 与 skill_state_debug / recorder 保持一致：基于 2560x1600 的偏移按窗口等比缩放
    scale = min(w / 2560, h / 1600)
    roi_offset_x = int(args.roi_offset_x * scale)
    roi_offset_y = int(args.roi_offset_y * scale)
    roi_w = max(1, int(args.roi_w * scale))
    roi_h = max(1, int(args.roi_h * scale))
    print(f"ROI 缩放系数: {scale:.3f}，实际偏移: ({roi_offset_x},{roi_offset_y})，实际尺寸: {roi_w}x{roi_h}")

    tile_calc = TilePosCalculator(
        screen_width=w,
        screen_height=h,
        grid_rows=rows,
        grid_cols=cols,
        stage_code=args.stage_code,
        stage_name=args.stage_name,
    )

    canvas = image.copy()
    if args.polygons:
        canvas = draw_tile_polygons(canvas, tile_calc, side=False, outline_color=(0, 0, 255), fill_color=(0, 0, 255))
        if args.side:
            canvas = draw_tile_polygons(canvas, tile_calc, side=True, outline_color=(255, 0, 0), fill_color=(255, 0, 0))

    canvas = draw_grid_positions(canvas, tile_calc, side=False, color=(0, 0, 255))
    if args.side:
        canvas = draw_grid_positions(canvas, tile_calc, side=True, color=(255, 0, 0))

    if args.deploy_row is not None and args.deploy_col is not None:
        if not (0 <= args.deploy_row < rows and 0 <= args.deploy_col < cols):
            print(f"错误：部署格子 ({args.deploy_row},{args.deploy_col}) 超出地图范围 {rows}x{cols}")
        else:
            deploy_x, deploy_y = tile_calc.get_screen_pos(args.deploy_row, args.deploy_col, side=False)
            canvas = draw_deploy_roi(
                canvas,
                deploy_x,
                deploy_y,
                roi_offset_x,
                roi_offset_y,
                roi_w,
                roi_h,
            )

    if args.output:
        output_path = Path(args.output)
    else:
        debug_dir = Path("debug") / "grid_positions"
        debug_dir.mkdir(parents=True, exist_ok=True)
        ts = int(time.time() * 1000)
        suffix = f"_{args.stage_code}" if args.stage_code else ""
        output_path = debug_dir / f"grid{suffix}_{rows}x{cols}_{ts}.png"

    cv2.imwrite(str(output_path), canvas)
    print(f"已保存: {output_path.absolute()}")
    print(f"使用地图尺寸: {rows}x{cols}")

    # 同时打印所有坐标供复制
    print("\nnormal view 坐标:")
    for r in range(tile_calc.grid_rows):
        row_positions = [str(tile_calc.get_screen_pos(r, c, side=False)) for c in range(tile_calc.grid_cols)]
        print(f"row {r}: {row_positions}")

    if args.side:
        print("\nside view 坐标:")
        for r in range(tile_calc.grid_rows):
            row_positions = [str(tile_calc.get_screen_pos(r, c, side=True)) for c in range(tile_calc.grid_cols)]
            print(f"row {r}: {row_positions}")


if __name__ == "__main__":
    main()
