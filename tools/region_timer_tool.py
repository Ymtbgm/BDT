import sys
import time
from pathlib import Path

import keyboard

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture
from core.region_state_timer import RegionStateTimer


def main():
    debug = "--debug" in sys.argv or "-debug" in sys.argv

    cap = WindowCapture(backend="mss")
    timer = RegionStateTimer(cap, debug=debug)

    print("[区域计时工具]")
    print("按 F9 开始/重置计时")
    print("按 F12 停止计时")
    print("按 ESC 退出")
    if debug:
        print("[DEBUG] 调试输出已开启")

    state = "idle"  # idle, running
    interval = 1.0 / 50.0  # 50Hz 检测，平衡精度和 tick 处理时间
    tick_start = time.perf_counter()

    while True:
        if keyboard.is_pressed("esc"):
            print("\n退出")
            timer.stop()
            break

        if keyboard.is_pressed("f9"):
            if state == "running":
                timer.start()
                timer.manual_pause()
                print("\n[工具] 计时已重置并暂停，按 F9 继续...")
            else:
                state = "running"
                timer.start(use_cost_detection=True)
                print("\n[工具] 等待费用条检测并启动计时...")
            time.sleep(0.5)
            tick_start = time.perf_counter()

        if keyboard.is_pressed("f12"):
            if state == "running":
                timer.stop()
                state = "idle"
                print(f"\n[工具] 计时停止，最终时间: {timer.get_elapsed_ms():.1f}ms")
            time.sleep(0.5)
            tick_start = time.perf_counter()

        if keyboard.is_pressed("f11") and state == "running":
            timer.toggle_manual_pause()
            print(f"\n[工具] 手动 {'暂停' if timer.is_manual_paused() else '继续'}")
            time.sleep(0.5)
            tick_start = time.perf_counter()

        if state == "running":
            loop_start = time.perf_counter()
            info = timer.tick()
            if info.get("started"):
                print(
                    f"\r时间: {info['elapsed_ms']:7.1f}ms | "
                    f"A={info['count_a']} B={info['count_b']} | "
                    f"倍率={info['rate']} | 暂停={info['paused']}",
                    end="",
                    flush=True,
                )
            if debug:
                d = (time.perf_counter() - loop_start) * 1000
                if d > interval * 1000:
                    print(f"\n[工具] tick 耗时 {d:.1f}ms，已无法维持 50fps")

        # 只在跑得过快时才等待；如果 tick 本身已经用掉了一帧以上，立即进入下一次
        elapsed = time.perf_counter() - tick_start
        sleep_time = max(0.0, interval - elapsed)
        if sleep_time > 0:
            time.sleep(sleep_time)
        tick_start = time.perf_counter()


if __name__ == "__main__":
    main()
