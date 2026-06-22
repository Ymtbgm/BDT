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


def _load_template(path: str) -> Optional[np.ndarray]:
    try:
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        return img
    except Exception:
        return None


def _capture_gray(cap: WindowCapture, roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    try:
        img = cap.capture_roi(*roi)
        if img.size == 0:
            return None
        if img.ndim == 3 and img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if img.ndim == 3 and img.shape[2] == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception:
        return None


def _template_score(gray: np.ndarray, tmpl: np.ndarray) -> float:
    if (
        tmpl is None
        or gray.shape[0] < tmpl.shape[0]
        or gray.shape[1] < tmpl.shape[1]
    ):
        return -1.0
    result = cv2.matchTemplate(gray, tmpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return float(max_val)


def _white_count(gray: np.ndarray, threshold: int = 200) -> int:
    return int(np.sum(gray > threshold))


def _frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """返回两帧的归一化互相关系数（1.0 表示完全相同）。"""
    if a.shape != b.shape:
        return 0.0
    a_f = a.astype(np.float32).flatten()
    b_f = b.astype(np.float32).flatten()
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    if denom == 0:
        return 1.0 if np.allclose(a_f, b_f) else 0.0
    return float(np.dot(a_f, b_f) / denom)


def main():
    save_frames = "--save" in sys.argv or "-save" in sys.argv

    cap = WindowCapture(backend="mss")
    roi_a = RegionStateTimer.DEFAULT_ROI_A

    # 加载与 RegionStateTimer 相同的模板
    templates = {}
    tmpl_root = ROOT / "core" / "resource"
    for name in ("time_run", "pause"):
        path = str(tmpl_root / f"{name}.png")
        tmpl = _load_template(path)
        if tmpl is not None:
            templates[name] = tmpl
            print(f"[模板] 已加载 {name}.png: {tmpl.shape}")
        else:
            print(f"[警告] 无法加载模板 {path}")

    out_dir = ROOT / "debug" / "region_capture_verify"
    if save_frames:
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"[输出] 帧将保存到: {out_dir}")

    print("\n[区域A连续截图验证工具]")
    print("按 F9 开始 1 秒 50 帧采集（请提前让游戏处于暂停态，采集期间按恢复键）")
    print("按 ESC 退出")

    interval = 1.0 / 50.0
    frames = []

    while True:
        if keyboard.is_pressed("esc"):
            print("\n退出")
            break

        if keyboard.is_pressed("f9"):
            print("\n[采集] 1 秒后开始...")
            time.sleep(1.0)
            print("[采集] 开始！在这 1 秒内进行暂停/恢复操作")

            frames.clear()
            start = time.perf_counter()
            prev_gray: Optional[np.ndarray] = None
            for i in range(50):
                t0 = time.perf_counter()
                gray = _capture_gray(cap, roi_a)
                ts = (time.perf_counter() - start) * 1000.0
                if gray is not None:
                    scores = {
                        name: _template_score(gray, tmpl)
                        for name, tmpl in templates.items()
                    }
                    best_state = max(scores, key=scores.get) if scores else None
                    best_score = scores.get(best_state, -1.0) if best_state else -1.0
                    white = _white_count(gray)
                    similarity = (
                        _frame_similarity(gray, prev_gray)
                        if prev_gray is not None
                        else 1.0
                    )
                    frames.append({
                        "idx": i,
                        "ts_ms": ts,
                        "white": white,
                        "scores": scores,
                        "best_state": best_state,
                        "best_score": best_score,
                        "similarity": similarity,
                        "gray": gray,
                    })
                    prev_gray = gray

                    if save_frames:
                        cv2.imencode(
                            ".png", gray
                        )[1].tofile(
                            str(out_dir / f"frame_{i:03d}_{ts:06.1f}ms.png")
                        )

                elapsed = time.perf_counter() - t0
                sleep_time = max(0.0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            duration = (time.perf_counter() - start) * 1000.0
            print(f"[采集] 完成，实际耗时 {duration:.1f}ms，共 {len(frames)} 帧\n")

            # 输出结果表格
            header = (
                f"{'帧':>3} | {'时间ms':>8} | {'白像素':>7} | "
                f"{'best':>6} | {'score':>6} | {'相似度':>6}"
            )
            print(header)
            print("-" * len(header))
            for f in frames:
                scores_str = ", ".join(
                    f"{k}={v:.2f}" for k, v in f["scores"].items()
                )
                print(
                    f"{f['idx']:>3} | {f['ts_ms']:>8.1f} | {f['white']:>7} | "
                    f"{str(f['best_state']):>6} | {f['best_score']:>6.2f} | "
                    f"{f['similarity']:>6.3f}  ({scores_str})"
                )

            # 简单分析：找出连续低分/None 的区间
            low_score_frames = [
                f for f in frames
                if f["best_score"] < 0.8
            ]
            if low_score_frames:
                print(
                    f"\n[分析] 有 {len(low_score_frames)} 帧最佳模板分数 < 0.8，"
                    f"可能是过渡帧"
                )
                for f in low_score_frames[:10]:
                    print(
                        f"  帧 {f['idx']} @ {f['ts_ms']:.1f}ms: best={f['best_state']} "
                        f"score={f['best_score']:.2f}, white={f['white']}"
                    )
            else:
                print("\n[分析] 所有帧都有明确的模板匹配（>=0.8）")

            print("\n按 F9 重新采集，ESC 退出")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
