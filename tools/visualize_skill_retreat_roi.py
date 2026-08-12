"""可视化录制器中技能和撤退按钮的 ROI 区域。

运行后截取当前游戏窗口，并在窗口截图上标出 ActionRecorder 使用的
SKILL（蓝色）和 RETREAT（红色）判定区域，便于排查点击未识别问题。

支持通过 --stage 传入关卡代号，使用 recorder 中基于 side view 投影锚点
计算的动态 ROI；否则使用旧的固定 ROI。
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture.capture import WindowCapture
from core.map.tile_pos import TilePosCalculator, load_stage_dimensions
from core.recording.recorder import ActionRecorder

# 基准分辨率
_BASE_W = 2560
_BASE_H = 1600

_COLORS = {
    "retreat": (0, 0, 255),  # 红色
    "skill": (255, 128, 0),  # 橙色
}


def _project_anchor_to_screen(
    calc: TilePosCalculator, wx: float, wy: float, wz: float
) -> tuple[float, float]:
    """把等效世界锚点通过 calc 的 side view 投影矩阵投到屏幕坐标。"""
    matrix = calc._get_transform_matrix(side=True)
    px, py, _, pw = np.dot(matrix, np.array([wx, wy, wz, 1.0]))
    sx = (1 + px / pw) / 2 * calc.screen_width
    sy = (1 - py / pw) / 2 * calc.screen_height
    return sx, sy


def _compute_roi_topleft(stage_code: str | None) -> dict[str, tuple[int, int, int, int]]:
    """返回 2560x1600 下技能和撤退 ROI 的左上角与宽高。

    如果提供 stage_code，使用 ActionRecorder 的投影锚点模型；
    否则回退到旧的固定 ROI。
    """
    retreat_w = ActionRecorder._RETREAT_W
    retreat_h = ActionRecorder._RETREAT_H
    skill_w = ActionRecorder._SKILL_W
    skill_h = ActionRecorder._SKILL_H

    if stage_code is None:
        return {
            "retreat": (1145, 510, retreat_w, retreat_h),
            "skill": (1615, 885, skill_w, skill_h),
        }

    try:
        dims = load_stage_dimensions(stage_code)
        if dims is None:
            raise ValueError(f"找不到关卡 '{stage_code}' 的尺寸信息")
        grid_cols, grid_rows = dims
        calc = TilePosCalculator(_BASE_W, _BASE_H, grid_rows, grid_cols, stage_code=stage_code)
        cx, cy = _project_anchor_to_screen(calc, *ActionRecorder._RETREAT_ANCHOR)
        sx, sy = _project_anchor_to_screen(calc, *ActionRecorder._SKILL_ANCHOR)
        return {
            "retreat": (int(round(cx - retreat_w / 2)), int(round(cy - retreat_h / 2)), retreat_w, retreat_h),
            "skill": (int(round(sx - skill_w / 2)), int(round(sy - skill_h / 2)), skill_w, skill_h),
        }
    except Exception as e:
        print(f"无法根据关卡 '{stage_code}' 计算动态 ROI: {e}")
        print("回退到固定 ROI")
        return {
            "retreat": (1145, 510, retreat_w, retreat_h),
            "skill": (1615, 885, skill_w, skill_h),
        }


def scale_roi(
    window_width: int,
    window_height: int,
    stage_code: str | None,
) -> dict[str, tuple[int, int, int, int]]:
    """将基于 2560x1600 的 ROI 缩放到当前窗口尺寸，返回窗口内相对坐标。"""
    sx = window_width / _BASE_W
    sy = window_height / _BASE_H
    base = _compute_roi_topleft(stage_code)
    return {
        name: (
            int(round(x * sx)),
            int(round(y * sy)),
            int(round(w * sx)),
            int(round(h * sy)),
        )
        for name, (x, y, w, h) in base.items()
    }


def draw_roi(
    canvas,
    x: int,
    y: int,
    w: int,
    h: int,
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
):
    h_img, w_img = canvas.shape[:2]
    x1 = max(0, min(w_img - 1, x))
    y1 = max(0, min(h_img - 1, y))
    x2 = max(0, min(w_img, x + w))
    y2 = max(0, min(h_img, y + h))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        canvas,
        label,
        (x1 + 2, max(y1 - 6, 18)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    size_text = f"{w}x{h}"
    (tw, th), _ = cv2.getTextSize(size_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(
        canvas,
        size_text,
        (x2 - tw - 4, y2 - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def _save_image(canvas, output_path: Path) -> bool:
    ok = cv2.imwrite(str(output_path), canvas)
    if not ok:
        try:
            encoded = cv2.imencode(".png", canvas)[1]
            output_path.write_bytes(encoded.tobytes())
            return True
        except Exception as e:
            print(f"imencode 也失败: {e}")
            return False
    return True


def main():
    parser = argparse.ArgumentParser(description="可视化技能和撤退按钮 ROI")
    parser.add_argument(
        "--window-title", type=str, default="明日方舟", help="游戏窗口标题，默认 明日方舟"
    )
    parser.add_argument(
        "--stage", "-s", type=str, help="关卡代号（如 7-18 / TO-9），使用动态拟合 ROI"
    )
    parser.add_argument(
        "--image", "-i", type=Path, help="从已有截图加载（ROI 会按图片尺寸缩放）"
    )
    parser.add_argument(
        "--output", "-o", type=Path, help="输出图片路径，默认 debug/visualize_skill_retreat_roi/<time>.png"
    )
    parser.add_argument(
        "--live", "-l", action="store_true", help="实时预览，按 Q 或 ESC 退出"
    )
    args = parser.parse_args()

    if args.image:
        if not args.image.exists():
            print(f"图片不存在: {args.image}")
            raise SystemExit(1)
        img = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"读取图片失败: {args.image}")
            raise SystemExit(1)
        window_height, window_width = img.shape[:2]
        print(f"从文件加载: {args.image}")
    else:
        cap = WindowCapture(window_title=args.window_title)
        window_width, window_height = cap.get_window_size()
        img = cap.capture()
        print(f"已截取游戏窗口: {window_width}x{window_height}")

    base_roi = _compute_roi_topleft(args.stage)
    print(f"基准 ROI (2560x1600):")
    for name, (x, y, w, h) in base_roi.items():
        print(f"  {name.upper()}: x={x}, y={y}, w={w}, h={h}")

    rois = scale_roi(window_width, window_height, args.stage)
    print(f"缩放后 ROI ({window_width}x{window_height}):")
    for name, (x, y, w, h) in rois.items():
        print(f"  {name.upper()}: x={x}, y={y}, w={w}, h={h}")

    canvas = img.copy()
    rx, ry, rw, rh = rois["retreat"]
    sx, sy, sw, sh = rois["skill"]
    draw_roi(canvas, rx, ry, rw, rh, _COLORS["retreat"], "RETREAT")
    draw_roi(canvas, sx, sy, sw, sh, _COLORS["skill"], "SKILL")

    if args.output is None:
        out_dir = ROOT / "debug" / "visualize_skill_retreat_roi"
        out_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{args.stage}" if args.stage else ""
        args.output = out_dir / f"{int(time.time() * 1000)}{suffix}_{window_width}x{window_height}.png"
    else:
        args.output = args.output.resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)

    if _save_image(canvas, args.output):
        print(f"已保存: {args.output}")
    else:
        print(f"保存失败: {args.output}")
        raise SystemExit(1)

    if args.live:
        print("实时预览中，按 Q 或 ESC 退出...")
        cap = WindowCapture(window_title=args.window_title)
        window_name = "Skill/Retreat ROI"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        while True:
            window_width, window_height = cap.get_window_size()
            frame = cap.capture()
            live_rois = scale_roi(window_width, window_height, args.stage)
            rx, ry, rw, rh = live_rois["retreat"]
            sx, sy, sw, sh = live_rois["skill"]
            draw_roi(frame, rx, ry, rw, rh, _COLORS["retreat"], "RETREAT")
            draw_roi(frame, sx, sy, sw, sh, _COLORS["skill"], "SKILL")
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(100) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
