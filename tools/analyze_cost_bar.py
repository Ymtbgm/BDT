import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def analyze_frames(frames_dir: Path, threshold: int = 200):
    """统计目录中所有费用条截图的白像素数量，并按数量排序输出。"""
    images = sorted(frames_dir.glob("frame_*.png"))
    if not images:
        print(f"目录中没有找到 frame_*.png: {frames_dir}")
        return

    print(f"共 {len(images)} 张图片，白像素阈值: {threshold}")
    print(f"{'文件名':<20} {'白像素数':>10} {'占比%':>8}")
    print("-" * 45)

    stats = []
    for path in images:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        white_count = int(np.sum(img > threshold))

        ratio = white_count / img.size * 100
        stats.append((path.name, white_count, ratio))
        print(f"{path.name:<20} {white_count:>10} {ratio:>8.2f}")

    print("\n按白像素数升序:")
    for name, count, ratio in sorted(stats, key=lambda x: x[1]):
        print(f"{name:<20} {count:>10} {ratio:>8.2f}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="分析费用条截图的白像素分布")
    parser.add_argument(
        "--dir",
        type=str,
        default=str(ROOT / "debug" / "cost_bar_frames"),
        help="费用条截图目录",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=200,
        help="白像素阈值（默认 200）",
    )
    args = parser.parse_args()

    analyze_frames(Path(args.dir), args.threshold)


if __name__ == "__main__":
    main()
