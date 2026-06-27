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

# 危机合约费用回复 tag：游戏实际仍为 30fps，但费用条完成一次回费循环
# 所需的“帧数”发生变化。这里记录每个 tag 下单次循环的帧数。
# 为了降低截图定时误差，在子弹时间下以 5 倍原间隔（166.67ms）截取，
# 对应正常游戏时间的一帧，但相对定时抖动被摊薄。
TAG_PRESETS = {
    "25": {"cycle_frames": 40, "label": "费用回复降低25%", "default_count": 80},
    "50": {"cycle_frames": 60, "label": "费用回复降低50%", "default_count": 120},
    "75": {"cycle_frames": 120, "label": "费用回复降低75%", "default_count": 240},
}

# 游戏实际运行帧率
GAME_FPS = 30.0
# 子弹时间倍数：截图间隔 = 正常一帧时长 * BULLET_TIME_FACTOR
BULLET_TIME_FACTOR = 5
CAPTURE_INTERVAL_MS = (1000.0 / GAME_FPS) * BULLET_TIME_FACTOR


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


def white_count(img_bgra: np.ndarray, threshold: int = 200) -> int:
    gray = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2GRAY)
    return int(np.sum(gray > threshold))


def save_frame(img, out_dir: Path, idx: int, threshold: int = 200):
    """保存截图并打印白像素统计信息。"""
    wc = white_count(img, threshold)
    total = img.shape[0] * img.shape[1]
    path = out_dir / f"frame_{idx:04d}.png"
    cv2.imencode(".png", img)[1].tofile(str(path))
    print(
        f"保存 {path.name}: 白像素(>{threshold})={wc}/{total} "
        f"({wc / total * 100:.1f}%)"
    )


def _sleep_until(target: float, spin_threshold: float = 0.005):
    """睡到目标时间附近，最后 spin_threshold 秒用忙等避免 Windows sleep 粒度误差。"""
    now = time.perf_counter()
    if now >= target:
        return
    sleep_until = target - spin_threshold
    if now < sleep_until:
        time.sleep(sleep_until - now)
    while time.perf_counter() < target:
        pass


def analyze_sequence(counts: list[int]):
    """根据白像素序列打印基本统计与相邻帧差值，辅助标定 step_pixels。"""
    if not counts:
        return
    deltas = [counts[i] - counts[i - 1] for i in range(1, len(counts))]
    positive_deltas = [d for d in deltas if d > 0]
    negative_deltas = [d for d in deltas if d < 0]
    print("\n=== 标定辅助统计 ===")
    print(f"总帧数: {len(counts)}")
    print(f"白像素范围: {min(counts)} ~ {max(counts)}")
    print(f"平均增量(正): {np.mean(positive_deltas):.2f}" if positive_deltas else "无正增量")
    print(f"最大单帧增量: {max(deltas) if deltas else 0}")
    print(f"最小单帧增量: {min(deltas) if deltas else 0}")
    print(f"归零/跳变次数(负增量): {len(negative_deltas)}")
    if negative_deltas:
        print(f"负增量均值: {np.mean(negative_deltas):.2f}")
    # 估算 step_pixels：相邻正增量的中位数
    if positive_deltas:
        step = float(np.median(positive_deltas))
        print(f"建议 step_pixels (中位数): {step:.2f}")
    print("====================\n")


def parse_args():
    parser = argparse.ArgumentParser(description="危机合约费用条标定截图工具")
    parser.add_argument(
        "--tag",
        choices=["25", "50", "75"],
        help="费用回复降低 tag（25/50/75），未指定时进入交互选择",
    )
    parser.add_argument("--count", type=int, help="截图帧数，默认按 tag 预设")
    parser.add_argument("--threshold", type=int, default=200, help="白像素灰度阈值，默认 200")
    parser.add_argument(
        "--out-dir", type=Path, help="输出目录，默认 debug/cost_bar_cc/{tag}"
    )
    return parser.parse_args()


def choose_tag() -> str:
    print("选择危机合约费用回复 tag:")
    for key, preset in TAG_PRESETS.items():
        print(f"  {key} - {preset['label']}（费用条循环约 {preset['cycle_frames']} 游戏帧）")
    while True:
        choice = input("输入 25/50/75: ").strip()
        if choice in TAG_PRESETS:
            return choice
        print("无效输入，请重新输入。")


def main():
    args = parse_args()
    tag = args.tag or choose_tag()
    preset = TAG_PRESETS[tag]
    cycle_frames = preset["cycle_frames"]
    interval_ms = CAPTURE_INTERVAL_MS
    count = args.count or preset["default_count"]
    out_dir = args.out_dir or (ROOT / "debug" / "cost_bar_cc" / tag)
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = WindowCapture(backend="mss")

    print(f"\n模式: {preset['label']}")
    print(f"子弹时间倍数: {BULLET_TIME_FACTOR}x")
    print(f"截图间隔: {interval_ms:.3f} ms（对应正常游戏 {1000.0/GAME_FPS:.3f} ms/帧）")
    print(f"费用条单循环约: {cycle_frames} 游戏帧")
    print(f"计划截图: {count} 帧（约 {count * interval_ms / 1000:.2f} 秒现实时间）")
    print(f"输出目录: {out_dir}")
    print("操作说明:")
    print("  F9    - 手动截取一帧")
    print(f"  P     - 自动连续截取 {count} 帧")
    print("  ESC   - 退出")
    print("提示：自动截图前请确保费用条处于刚归零或刚满条后的稳定递增阶段，便于标定。\n")

    frame_idx = 0
    counts = []
    while True:
        if keyboard.is_pressed("f9"):
            try:
                img = capture_cost_bar_roi(cap)
                save_frame(img, out_dir, frame_idx, args.threshold)
                counts.append(white_count(img, args.threshold))
                frame_idx += 1
            except Exception as e:
                print(f"截图失败: {e}")
            time.sleep(0.3)

        elif keyboard.is_pressed("p"):
            print(f"开始自动连续截取 {count} 帧，请保持游戏画面稳定...")
            start = time.perf_counter()
            for i in range(count):
                target = start + (i + 1) * interval_ms / 1000.0
                try:
                    img = capture_cost_bar_roi(cap)
                    save_frame(img, out_dir, frame_idx, args.threshold)
                    counts.append(white_count(img, args.threshold))
                    frame_idx += 1
                except Exception as e:
                    print(f"第 {i + 1}/{count} 帧截图失败: {e}")
                _sleep_until(target)
            print("自动截取完成")
            analyze_sequence(counts[-count:])
            time.sleep(0.5)

        elif keyboard.is_pressed("esc"):
            print("退出")
            break

        time.sleep(0.05)


if __name__ == "__main__":
    main()
