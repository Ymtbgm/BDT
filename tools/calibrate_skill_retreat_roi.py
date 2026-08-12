"""交互式标定技能和撤退按钮中心位置。

启动后：
- 把鼠标放到技能按钮中心，按 F9 记录技能位置
- 把鼠标放到撤退按钮中心，再按 F9 记录撤退位置
- 按 ESC 或 Q 退出并保存

保存内容：
- debug/calibrate_skill_retreat_roi/<timestamp>/annotated.png
- debug/calibrate_skill_retreat_roi/<timestamp>/positions.json
"""

import json
import sys
import time
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import win32gui
from pynput import keyboard

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture.capture import WindowCapture

# 与 core/recording/recorder.py 保持一致
_BASE_W = 2560
_BASE_H = 1600
_RETREAT_W = 170
_RETREAT_H = 160
_SKILL_W = 250
_SKILL_H = 200

_COLORS = {
    "retreat": (0, 0, 255),  # 红色
    "skill": (255, 128, 0),  # 橙色（更醒目）
}


def get_mouse_window_pos(monitor: dict) -> tuple[int, int]:
    """返回鼠标相对于窗口左上角的坐标（物理像素）。"""
    abs_x, abs_y = win32gui.GetCursorPos()
    return abs_x - monitor["left"], abs_y - monitor["top"]


def roi_from_center(
    cx: int, cy: int, w: int, h: int, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """根据中心点和宽高计算 ROI，限制在图片范围内。"""
    x1 = max(0, cx - w // 2)
    y1 = max(0, cy - h // 2)
    x2 = min(img_w, x1 + w)
    y2 = min(img_h, y1 + h)
    # 如果一侧被截断，尝试向另一侧对齐
    if x2 - x1 < w and x1 > 0:
        x1 = max(0, x2 - w)
    if y2 - y1 < h and y1 > 0:
        y1 = max(0, y2 - h)
    return x1, y1, x2, y2


def draw_roi(
    canvas: np.ndarray,
    roi: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    x1, y1, x2, y2 = roi
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


def draw_crosshair(canvas: np.ndarray, x: int, y: int, color: tuple[int, int, int], size: int = 20):
    cv2.line(canvas, (x - size, y), (x + size, y), color, 2)
    cv2.line(canvas, (x, y - size), (x, y + size), color, 2)
    cv2.circle(canvas, (x, y), size // 2, color, 1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="标定技能和撤退按钮中心位置")
    parser.add_argument("--window-title", type=str, default="明日方舟", help="游戏窗口标题")
    args = parser.parse_args()

    cap = WindowCapture(window_title=args.window_title)
    monitor = cap.monitor
    window_width, window_height = cap.get_window_size()
    scale_x = window_width / _BASE_W
    scale_y = window_height / _BASE_H

    print(f"窗口尺寸: {window_width}x{window_height}")
    print("操作说明:")
    print("  第 1 次按 F9：记录技能按钮中心（鼠标所在位置）")
    print("  第 2 次按 F9：记录撤退按钮中心（鼠标所在位置）")
    print("  按 ESC 或 Q：保存并退出")

    skill_center: tuple[int, int] | None = None
    retreat_center: tuple[int, int] | None = None
    latest_frame: np.ndarray | None = None
    lock = Lock()
    stop_flag = False

    def on_press(key):
        nonlocal skill_center, retreat_center, stop_flag
        if key == keyboard.Key.esc:
            stop_flag = True
            return False
        if key == keyboard.Key.f9:
            rel_x, rel_y = get_mouse_window_pos(monitor)
            with lock:
                if skill_center is None:
                    skill_center = (rel_x, rel_y)
                    print(f"[技能] 记录中心: ({rel_x}, {rel_y})")
                elif retreat_center is None:
                    retreat_center = (rel_x, rel_y)
                    print(f"[撤退] 记录中心: ({rel_x}, {rel_y})")
                else:
                    print("两个位置都已记录，按 ESC 或 Q 保存退出")
        try:
            if key.char and key.char.lower() == "q":
                stop_flag = True
                return False
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    window_name = "Skill/Retreat ROI Calibration"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while not stop_flag:
            try:
                frame = cap.capture()
            except Exception as e:
                print(f"截图失败: {e}")
                time.sleep(0.1)
                continue

            with lock:
                canvas = frame.copy()
                skill_roi = None
                retreat_roi = None
                if skill_center is not None:
                    skill_w = int(round(_SKILL_W * scale_x))
                    skill_h = int(round(_SKILL_H * scale_y))
                    skill_roi = roi_from_center(
                        skill_center[0], skill_center[1], skill_w, skill_h,
                        window_width, window_height
                    )
                    draw_crosshair(canvas, skill_center[0], skill_center[1], _COLORS["skill"])
                    draw_roi(canvas, skill_roi, _COLORS["skill"], "SKILL")
                if retreat_center is not None:
                    retreat_w = int(round(_RETREAT_W * scale_x))
                    retreat_h = int(round(_RETREAT_H * scale_y))
                    retreat_roi = roi_from_center(
                        retreat_center[0], retreat_center[1], retreat_w, retreat_h,
                        window_width, window_height
                    )
                    draw_crosshair(canvas, retreat_center[0], retreat_center[1], _COLORS["retreat"])
                    draw_roi(canvas, retreat_roi, _COLORS["retreat"], "RETREAT")

                # 显示操作提示
                status_lines = [
                    "F9: record SKILL -> RETREAT | ESC/Q: save & exit",
                ]
                if skill_center is None:
                    status_lines.append("Waiting for SKILL center...")
                elif retreat_center is None:
                    status_lines.append("Waiting for RETREAT center...")
                else:
                    status_lines.append("Both recorded. Press ESC/Q to save.")

                y_offset = 30
                for line in status_lines:
                    cv2.putText(
                        canvas, line, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA,
                    )
                    y_offset += 25

                latest_frame = canvas.copy()

            # 缩放显示，避免窗口太大
            display_scale = min(1.0, 1400 / max(window_width, window_height))
            if display_scale < 1.0:
                display_w = int(window_width * display_scale)
                display_h = int(window_height * display_scale)
                display_img = cv2.resize(canvas, (display_w, display_h))
            else:
                display_img = canvas

            cv2.imshow(window_name, display_img)
            if cv2.waitKey(50) & 0xFF == 27:  # ESC
                stop_flag = True
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                stop_flag = True
                break
    finally:
        listener.stop()
        cv2.destroyAllWindows()

    # 保存结果
    if latest_frame is not None:
        out_dir = ROOT / "debug" / "calibrate_skill_retreat_roi" / str(int(time.time() * 1000))
        out_dir.mkdir(parents=True, exist_ok=True)

        img_path = out_dir / "annotated.png"
        encoded = cv2.imencode(".png", latest_frame)[1]
        img_path.write_bytes(encoded.tobytes())
        print(f"已保存截图: {img_path}")

        positions = {
            "window_size": {"width": window_width, "height": window_height},
            "scale": {"x": scale_x, "y": scale_y},
            "skill": None,
            "retreat": None,
        }
        if skill_center is not None:
            positions["skill"] = {
                "window_center": {"x": skill_center[0], "y": skill_center[1]},
                "base_center": {
                    "x": round(skill_center[0] / scale_x, 2),
                    "y": round(skill_center[1] / scale_y, 2),
                },
            }
        if retreat_center is not None:
            positions["retreat"] = {
                "window_center": {"x": retreat_center[0], "y": retreat_center[1]},
                "base_center": {
                    "x": round(retreat_center[0] / scale_x, 2),
                    "y": round(retreat_center[1] / scale_y, 2),
                },
            }

        json_path = out_dir / "positions.json"
        json_path.write_text(json.dumps(positions, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"已保存位置: {json_path}")
    else:
        print("未截取到任何画面，未保存。")


if __name__ == "__main__":
    main()
