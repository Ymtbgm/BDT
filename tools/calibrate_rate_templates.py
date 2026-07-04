import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import keyboard
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture
from core.region_state_timer import RegionStateTimer


def _load_template(path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """加载模板，返回 (灰度图, 掩膜)。

    若原图含 alpha 通道，则使用 alpha 作为掩膜；否则按灰度阈值生成掩膜。
    """
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None, None
        if img.ndim == 3 and img.shape[2] == 4:
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            mask = (img[:, :, 3] > 128).astype(np.uint8) * 255
        elif img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        else:
            gray = img
            _, mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return gray, mask
    except Exception:
        return None, None


def _binarize(gray: np.ndarray, threshold: int) -> np.ndarray:
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary


def _match_score(
    roi: np.ndarray,
    tmpl: Optional[np.ndarray],
    mask: Optional[np.ndarray] = None,
    method: int = cv2.TM_CCOEFF_NORMED,
) -> float:
    if tmpl is None or roi.shape[0] < tmpl.shape[0] or roi.shape[1] < tmpl.shape[1]:
        return 0.0
    try:
        result = cv2.matchTemplate(roi, tmpl, method, mask=mask)
    except Exception:
        return 0.0
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def _frame_diff(prev: Optional[np.ndarray], curr: np.ndarray) -> float:
    if prev is None or prev.shape != curr.shape:
        return 0.0
    return float(np.mean(cv2.absdiff(prev, curr)))


def _decide_state(
    score_fast: float,
    score_slow: float,
    mean_diff: float,
    match_confidence: float,
    diff_threshold: float,
) -> str:
    if (
        score_fast >= match_confidence
        and score_fast > score_slow
        and mean_diff <= diff_threshold
    ):
        return "fast"
    if (
        score_slow >= match_confidence
        and score_slow > score_fast
        and mean_diff <= diff_threshold
    ):
        return "slow"
    return "transition"


def _burst_capture(
    cap: WindowCapture,
    roi: Tuple[int, int, int, int],
    tmpl_fast: Optional[np.ndarray],
    mask_fast: Optional[np.ndarray],
    tmpl_slow: Optional[np.ndarray],
    mask_slow: Optional[np.ndarray],
    match_confidence: float,
    diff_threshold: float,
    preprocess: str = "none",
    binary_threshold: int = 150,
    save: bool = True,
):
    """按下 F9 后采集 1 秒（30 帧，每 33ms 一帧）并输出状态表。"""
    print("\n[采集] 开始 1 秒 30 帧采集（33ms/帧），请在采集期间进行倍率切换...")

    method = cv2.TM_CCORR_NORMED if preprocess == "mask" else cv2.TM_CCOEFF_NORMED
    tmpl_fast_p = _binarize(tmpl_fast, binary_threshold) if preprocess == "binary" and tmpl_fast is not None else tmpl_fast
    tmpl_slow_p = _binarize(tmpl_slow, binary_threshold) if preprocess == "binary" and tmpl_slow is not None else tmpl_slow

    interval = 1.0 / 30.0
    frames = []
    prev_gray: Optional[np.ndarray] = None
    start = time.perf_counter()
    for i in range(30):
        t0 = time.perf_counter()
        img = cap.capture_roi(*roi)
        ts = (t0 - start) * 1000.0
        if img.size == 0:
            time.sleep(interval)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if preprocess == "binary":
            gray = _binarize(gray, binary_threshold)

        score_fast = _match_score(gray, tmpl_fast_p, mask=mask_fast if preprocess == "mask" else None, method=method)
        score_slow = _match_score(gray, tmpl_slow_p, mask=mask_slow if preprocess == "mask" else None, method=method)
        mean_diff = _frame_diff(prev_gray, gray)
        prev_gray = gray.copy()
        state = _decide_state(score_fast, score_slow, mean_diff, match_confidence, diff_threshold)

        frames.append({
            "idx": i,
            "ts_ms": ts,
            "score_fast": score_fast,
            "score_slow": score_slow,
            "diff": mean_diff,
            "state": state,
            "gray": gray,
        })

        # 保存帧图
        if save:
            out_dir = ROOT / "debug" / "calibrate_rate"
            out_dir.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", gray)[1].tofile(
                str(out_dir / f"burst_{int(start*1000)}_{i:03d}_{state}.png")
            )

        # 按 33ms 节拍等待
        elapsed = time.perf_counter() - t0
        sleep_time = max(0.0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)

    duration = (time.perf_counter() - start) * 1000.0
    print(f"[采集] 完成，实际耗时 {duration:.1f}ms，共 {len(frames)} 帧")
    if save:
        print(f"[采集] 帧图已保存到: {ROOT / 'debug' / 'calibrate_rate'}")

    # 输出结果表
    header = f"{'帧':>3} | {'时间ms':>8} | {'fast':>6} | {'slow':>6} | {'diff':>7} | state"
    print(header)
    print("-" * len(header))
    for f in frames:
        print(
            f"{f['idx']:>3} | {f['ts_ms']:>8.1f} | "
            f"{f['score_fast']:>6.3f} | {f['score_slow']:>6.3f} | "
            f"{f['diff']:>7.2f} | {f['state']}"
        )

    transitions = [f for f in frames if f["state"] == "transition"]
    if transitions:
        print(f"\n[分析] 共 {len(transitions)} 帧判定为 transition")
    else:
        print("\n[分析] 未检测到 transition 帧")
    print("\n按 F9 重新采集，ESC 退出")


def main():
    parser = argparse.ArgumentParser(description="倍率模板校准工具")
    parser.add_argument(
        "--preprocess",
        choices=["none", "mask", "binary"],
        default="none",
        help="预处理：none=原图, mask=模板掩膜, binary=二值化",
    )
    parser.add_argument(
        "--binary-threshold",
        type=int,
        default=150,
        help="二值化/掩膜阈值（默认 150）",
    )
    args = parser.parse_args()

    cap = WindowCapture(backend="mss")
    roi = RegionStateTimer.DEFAULT_ROI_B

    tmpl_root = ROOT / "core" / "resource"
    tmpl_fast, mask_fast = _load_template(str(tmpl_root / "1X.png"))
    tmpl_slow, mask_slow = _load_template(str(tmpl_root / "0.2X.png"))

    if tmpl_fast is None:
        print("[警告] 未找到 1X.png 模板")
    if tmpl_slow is None:
        print("[警告] 未找到 0.2X.png 模板，请用 tools/capture_region_templates.py F10 捕获")
    if tmpl_fast is None or tmpl_slow is None:
        print("按 ESC 退出")
        while not keyboard.is_pressed("esc"):
            time.sleep(0.05)
        return

    match_confidence = 0.85
    diff_threshold = 3.0

    method = cv2.TM_CCORR_NORMED if args.preprocess == "mask" else cv2.TM_CCOEFF_NORMED
    tmpl_fast_p = _binarize(tmpl_fast, args.binary_threshold) if args.preprocess == "binary" else tmpl_fast
    tmpl_slow_p = _binarize(tmpl_slow, args.binary_threshold) if args.preprocess == "binary" else tmpl_slow

    print("[倍率模板校准工具]")
    print(f"预处理: {args.preprocess}")
    print("持续捕获区域 B，实时显示 1x/0.2x 模板得分、帧差和判定状态")
    print("按 F9 开始 1 秒 30 帧采集（33ms/帧），按 ESC 退出")
    print()

    prev_gray: Optional[np.ndarray] = None
    last_state = "?"
    while True:
        if keyboard.is_pressed("esc"):
            print("\n退出")
            break
        if keyboard.is_pressed("f9"):
            _burst_capture(
                cap,
                roi,
                tmpl_fast,
                mask_fast,
                tmpl_slow,
                mask_slow,
                match_confidence,
                diff_threshold,
                preprocess=args.preprocess,
                binary_threshold=args.binary_threshold,
            )
            time.sleep(0.5)
            continue

        img = cap.capture_roi(*roi)
        if img.size == 0:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if args.preprocess == "binary":
            gray = _binarize(gray, args.binary_threshold)

        score_fast = _match_score(
            gray,
            tmpl_fast_p,
            mask=mask_fast if args.preprocess == "mask" else None,
            method=method,
        )
        score_slow = _match_score(
            gray,
            tmpl_slow_p,
            mask=mask_slow if args.preprocess == "mask" else None,
            method=method,
        )
        mean_diff = _frame_diff(prev_gray, gray)
        prev_gray = gray.copy()
        state = _decide_state(score_fast, score_slow, mean_diff, match_confidence, diff_threshold)

        if state != last_state:
            print(f"状态切换 -> {state}")
            last_state = state

        print(
            f"\rfast={score_fast:.3f} slow={score_slow:.3f} diff={mean_diff:6.2f} state={state}  ",
            end="",
            flush=True,
        )
        time.sleep(0.05)


if __name__ == "__main__":
    main()
