"""Benchmark mss ROI capture time for different region sizes.

Assumes a smaller Region B ROI is used and measures raw capture overhead
(converted to numpy array, as RegionStateTimer does).
"""
import time
import statistics

import mss
import numpy as np


SIZES = [
    (128, 119),
    (96, 96),
    (80, 80),
    (64, 64),
    (48, 48),
    (32, 32),
]

ITERATIONS = 200
WARMUP = 30


def benchmark(size, include_conversion: bool):
    w, h = size
    with mss.mss() as sct:
        monitor = {"left": 100, "top": 100, "width": w, "height": h}

        # Warmup
        for _ in range(WARMUP):
            img = sct.grab(monitor)
            if include_conversion:
                _ = np.array(img)

        times = []
        for _ in range(ITERATIONS):
            t0 = time.perf_counter()
            img = sct.grab(monitor)
            if include_conversion:
                arr = np.array(img)
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)
            if include_conversion:
                _ = arr.shape

    return {
        "avg": statistics.mean(times),
        "min": min(times),
        "max": max(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(ITERATIONS * 0.95)],
    }


def main():
    print(f"Benchmarking ROI capture ({ITERATIONS} iterations each)\n")
    print(f"{'Size':>10} {'mode':>14} {'avg_ms':>8} {'median':>8} {'min_ms':>8} {'max_ms':>8} {'p95_ms':>8}")
    for size in SIZES:
        for mode, include in [("grab+np", True), ("grab only", False)]:
            result = benchmark(size, include)
            print(
                f"{f'{size[0]}x{size[1]}':>10} "
                f"{mode:>14} "
                f"{result['avg']:>8.3f} "
                f"{result['median']:>8.3f} "
                f"{result['min']:>8.3f} "
                f"{result['max']:>8.3f} "
                f"{result['p95']:>8.3f}"
            )


if __name__ == "__main__":
    main()
