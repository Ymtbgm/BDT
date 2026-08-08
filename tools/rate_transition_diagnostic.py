"""倍率切换诊断工具。

按下配置的开始键（默认空格）后，在 2 秒内以高帧率连续截图区域 B，并同步记录鼠标状态
（左键是否按下、左键 mouseUp 时间戳）、倍率键（默认 F）按下/松开时间，以及开始键自身的时间戳。
按指定热键可在当前鼠标位置触发一次划火柴选中
（P 键按下 -> 左键点击 -> ESC 键松开），无需额外启动进程。

用法：
    python tools/rate_transition_diagnostic.py

输出：
    debug/rate_transition_diag/<timestamp>/
        frame_0000.png ...
        log.json
"""

import json
import signal
import sys
import time
from collections import deque
from pathlib import Path
from threading import Lock
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pynput.keyboard import Listener as KeyboardListener, Key
from pynput.mouse import Listener, Button, Controller as MouseController

_MOUSE_CONTROLLER = MouseController()
_active_capture: Optional[WindowCapture] = None


def _cleanup_diag(*_):
    print("\n[清理] 释放捕获资源...")
    global _active_capture
    if _active_capture is not None:
        _active_capture.stop()
    sys.exit(0)


signal.signal(signal.SIGINT, _cleanup_diag)
signal.signal(signal.SIGTERM, _cleanup_diag)

try:
    import pydirectinput
    pydirectinput.PAUSE = 0.0
except Exception:
    pydirectinput = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.capture import WindowCapture
from core.cost_bar_sync import CostBarSync
from core.region_state_timer import _RateTemplateMatcher, RegionStateTimer
from core import constants


from core.paths import GAME_TEMPLATE_DIR

# 区域 B 默认 ROI 与 RegionStateTimer 一致
ROI_B = RegionStateTimer.DEFAULT_ROI_B
FPS = 60
DURATION_S = 2.0
TEMPLATE_DIR = GAME_TEMPLATE_DIR


def _parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="倍率切换诊断工具")
    parser.add_argument(
        "--fps",
        type=int,
        default=120,
        help="采样帧率（默认 120fps）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="记录时长（秒，默认 2.0）",
    )
    parser.add_argument(
        "--matchstick-delay",
        type=float,
        default=100.0,
        help="划火柴中 ESC 按下后到松开的等待时间（ms，默认 100）",
    )
    parser.add_argument(
        "--start-key",
        type=str,
        default="space",
        help="开始/停止记录的热键（默认 space）",
    )
    parser.add_argument(
        "--exit-key",
        type=str,
        default="f10",
        help="退出程序的热键（默认 f10）",
    )
    parser.add_argument(
        "--matchstick-key",
        type=str,
        default="f8",
        help="划火柴选中热键（默认 f8）",
    )
    parser.add_argument(
        "--rate-key",
        type=str,
        default="f",
        help="游戏内倍率切换键（默认 f）",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mss",
        choices=["mss", "windows_capture"],
        help="截图后端（默认 mss，高刷新率可试 windows_capture）",
    )
    parser.add_argument(
        "--wgc-interval",
        type=int,
        default=6,
        help="windows-capture 最小更新间隔 ms（默认 6，165Hz 推荐 6）",
    )
    return parser.parse_args()


class RateTransitionDiagnostic:
    def __init__(
        self,
        matchstick_delay_ms: float = 100.0,
        matchstick_key: str = "f8",
        rate_key: str = "f",
        start_key: str = "space",
        exit_key: str = "f10",
        backend: str = "mss",
        wgc_interval: int = 6,
    ):
        self._lock = Lock()
        self._left_down = False
        self._mouse_downs: Deque[Tuple[float, int, int]] = deque()
        self._mouse_ups: Deque[Tuple[float, int, int]] = deque()
        self._matchstick_triggers: Deque[Tuple[float, int, int]] = deque()
        self._rate_key_downs: Deque[Tuple[float, str]] = deque()
        self._rate_key_ups: Deque[Tuple[float, str]] = deque()
        self._start_key_events: Deque[Tuple[float, str, bool]] = deque()
        self._matchstick_delay_ms = matchstick_delay_ms
        self._matchstick_key = matchstick_key.lower()
        self._rate_key = rate_key.lower()
        self._start_key = start_key.lower()
        self._exit_key = exit_key.lower()
        self._backend = backend
        self._wgc_interval = wgc_interval
        self._running = True
        self._start_pressed = False
        self._exit_pressed = False
        self._matchstick_key_pressed = False
        self._keyboard_listener: Optional[KeyboardListener] = None
        self._mouse_listener: Optional[Listener] = None

    def _on_control_key(self, key, pressed: bool):
        """处理控制热键 开始/退出 与划火柴热键。"""
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            if name == self._start_key:
                self._start_pressed = pressed
                self._start_key_events.append((time.perf_counter(), name, pressed))
            elif name == self._exit_key:
                self._exit_pressed = pressed
            elif name == self._matchstick_key:
                if pressed and not self._matchstick_key_pressed:
                    self._matchstick_key_pressed = True
                elif not pressed:
                    self._matchstick_key_pressed = False

    def _on_key_press(self, key):
        self._on_control_key(key, True)
        name = self._key_name(key)
        if name is None:
            return
        # 记录倍率键按下
        if name == self._rate_key:
            x, y = _MOUSE_CONTROLLER.position
            with self._lock:
                self._rate_key_downs.append((time.perf_counter(), name))
        # 划火柴热键在按下时触发（与 action.py 的 pynput 热键行为一致）
        if name is not None and name == self._matchstick_key:
            x, y = _MOUSE_CONTROLLER.position
            with self._lock:
                self._matchstick_triggers.append((time.perf_counter(), x, y))
            self._trigger_matchstick()

    def _on_key_release(self, key):
        self._on_control_key(key, False)
        name = self._key_name(key)
        if name is not None and name == self._rate_key:
            with self._lock:
                self._rate_key_ups.append((time.perf_counter(), name))

    @staticmethod
    def _key_name(key):
        if hasattr(key, "name") and key.name is not None:
            return key.name.lower()
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        return None

    def _start_listeners(self):
        self._mouse_listener = Listener(on_click=self._on_click)
        self._mouse_listener.start()
        self._keyboard_listener = KeyboardListener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
        )
        self._keyboard_listener.start()

    def _stop_listeners(self):
        if self._mouse_listener is not None:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def _on_click(self, x: int, y: int, button, pressed: bool):
        if button == Button.left:
            with self._lock:
                if pressed:
                    self._left_down = True
                    self._mouse_downs.append((time.perf_counter(), x, y))
                else:
                    self._left_down = False
                    self._mouse_ups.append((time.perf_counter(), x, y))
        elif button == Button.right and not pressed:
            # 右键松开：仅作为普通操作记录点，不触发划火柴
            pass

    def _flush_mouse_downs(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._mouse_downs and self._mouse_downs[0][0] <= until:
                ts, x, y = self._mouse_downs.popleft()
                if ts >= since:
                    events.append({"ts": ts, "x": x, "y": y})
            return events

    def _trigger_matchstick(self):
        """划火柴选中：在当前鼠标位置暂停+左键+恢复。"""
        if pydirectinput is None:
            print("[警告] 未安装 pydirectinput，无法执行划火柴")
            return
        pause_key = constants.DEFAULT_PAUSE_KEY
        pydirectinput.keyDown(pause_key)
        time.sleep(0.01)
        pydirectinput.click(button="left")
        pydirectinput.keyDown("esc")
        time.sleep(self._matchstick_delay_ms / 1000.0)
        pydirectinput.keyUp("esc")
        pydirectinput.keyUp(pause_key)

    def _flush_matchstick_triggers(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._matchstick_triggers and self._matchstick_triggers[0][0] <= until:
                ts, x, y = self._matchstick_triggers.popleft()
                if ts >= since:
                    events.append({"ts": ts, "x": x, "y": y})
            return events

    def _flush_mouse_ups(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._mouse_ups and self._mouse_ups[0][0] <= until:
                ts, x, y = self._mouse_ups.popleft()
                if ts >= since:
                    events.append({"ts": ts, "x": x, "y": y})
            return events

    def _flush_rate_key_downs(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._rate_key_downs and self._rate_key_downs[0][0] <= until:
                ts, name = self._rate_key_downs.popleft()
                if ts >= since:
                    events.append({"ts": ts, "name": name})
            return events

    def _flush_rate_key_ups(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._rate_key_ups and self._rate_key_ups[0][0] <= until:
                ts, name = self._rate_key_ups.popleft()
                if ts >= since:
                    events.append({"ts": ts, "name": name})
            return events

    def _flush_start_key_events(self, since: float, until: float) -> List[Dict]:
        with self._lock:
            events = []
            while self._start_key_events and self._start_key_events[0][0] <= until:
                ts, name, pressed = self._start_key_events.popleft()
                if ts >= since:
                    events.append({"ts": ts, "name": name, "pressed": pressed})
            return events

    def _is_left_down(self) -> bool:
        with self._lock:
            return self._left_down

    def run(self, fps: int = FPS, duration_s: float = DURATION_S):
        global _active_capture
        print("[倍率切换诊断工具]")
        print(f"后端: {self._backend}")
        print(f"ROI: {ROI_B}")
        print(f"采集: {fps}fps x {duration_s}s = {int(fps * duration_s)} 帧")
        print(f"按 {self._start_key.upper()} 开始记录，{self._exit_key.upper()} 退出，{self._matchstick_key.upper()} 在当前位置划火柴选中（P+左键+ESC），{self._rate_key.upper()} 记录倍率键")

        try:
            cap = WindowCapture(backend=self._backend, minimum_update_interval=self._wgc_interval)
            _active_capture = cap
        except Exception as e:
            print(f"[错误] 创建 WindowCapture 失败: {e}")
            return

        matcher = _RateTemplateMatcher(
            fast_path=str(TEMPLATE_DIR / constants.RATE_TEMPLATE_FAST_NAME),
            slow_path=str(TEMPLATE_DIR / constants.RATE_TEMPLATE_SLOW_NAME),
            fast2x_path=str(TEMPLATE_DIR / constants.RATE_TEMPLATE_FAST2X_NAME),
            match_confidence=constants.RATE_TEMPLATE_MATCH_CONFIDENCE,
            transition_confidence=constants.RATE_TEMPLATE_TRANSITION_CONFIDENCE,
            diff_threshold=constants.RATE_TEMPLATE_DIFF_THRESHOLD,
        )

        if not matcher.available:
            print("[错误] 倍率模板加载失败，请确认 resource/game_template/ 下有 1X.png / 2X.png / 0.2X.png")
            return

        cost_sync = CostBarSync(cap, debug=False)

        self._start_listeners()

        try:
            while self._running:
                with self._lock:
                    exit_pressed = self._exit_pressed
                    start_pressed = self._start_pressed

                if exit_pressed:
                    print("\n退出")
                    break

                if start_pressed:
                    self._record(cap, matcher, cost_sync, fps=fps, duration_s=duration_s)
                    # 等待开始键松开，防止长按触发多次
                    while True:
                        with self._lock:
                            if not self._start_pressed:
                                break
                        time.sleep(0.05)
                    continue

                time.sleep(0.05)
        finally:
            self._running = False
            self._stop_listeners()
            cap.stop()
            _active_capture = None

    def _record(self, cap: WindowCapture, matcher: _RateTemplateMatcher, cost_sync: CostBarSync, fps: int, duration_s: float):
        frame_count = int(fps * duration_s)
        interval = 1.0 / fps

        session_ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / "debug" / "rate_transition_diag" / session_ts
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[记录] 开始采集{duration_s}秒，输出目录: {out_dir}")
        # 给用户一点准备时间
        time.sleep(0.1)
        print("[记录] 开始！请进行倍率切换操作。")

        frames: List[Dict] = []
        start = time.perf_counter()
        prev_ts = start

        for i in range(frame_count):
            t0 = time.perf_counter()
            img = cap.capture_roi(*ROI_B)
            capture_end = time.perf_counter()

            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            score_fast, score_slow, score_fast2x, mean_diff, state = matcher.match(gray)

            # 同步截取费用条并统计白像素
            cost_gray = cost_sync.capture_roi_gray()
            cost_white = cost_sync.white_pixel_count(cost_gray) if cost_gray is not None else None

            # 取 capture 开始时刻作为帧时间戳
            frame_ts = t0
            # 收集本帧间隔内的 mouseDown、mouseUp 事件与划火柴触发
            mouse_downs = self._flush_mouse_downs(prev_ts, frame_ts)
            mouse_ups = self._flush_mouse_ups(prev_ts, frame_ts)
            matchstick_triggers = self._flush_matchstick_triggers(prev_ts, frame_ts)
            rate_key_downs = self._flush_rate_key_downs(prev_ts, frame_ts)
            rate_key_ups = self._flush_rate_key_ups(prev_ts, frame_ts)
            start_key_events = self._flush_start_key_events(prev_ts, frame_ts)
            left_down = self._is_left_down()

            frame_filename = f"frame_{i:04d}.png"
            cv2.imencode(".png", gray)[1].tofile(str(out_dir / frame_filename))

            frames.append({
                "idx": i,
                "ts": frame_ts,
                "ts_rel_ms": (frame_ts - start) * 1000.0,
                "capture_ms": (capture_end - t0) * 1000.0,
                "score_fast": score_fast,
                "score_slow": score_slow,
                "score_fast2x": score_fast2x,
                "mean_diff": mean_diff,
                "state": state,
                "left_down": left_down,
                "cost_white": cost_white,
                "mouse_downs_in_frame": mouse_downs,
                "mouse_ups_in_frame": mouse_ups,
                "matchstick_triggers_in_frame": matchstick_triggers,
                "rate_key_downs_in_frame": rate_key_downs,
                "rate_key_ups_in_frame": rate_key_ups,
                "start_key_events_in_frame": start_key_events,
                "frame_file": frame_filename,
            })

            prev_ts = frame_ts

            # 维持目标帧率
            elapsed = time.perf_counter() - t0
            sleep_time = max(0.0, interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = (time.perf_counter() - start) * 1000.0
        print(f"[记录] 完成，实际耗时 {duration:.1f}ms，共 {len(frames)} 帧")

        # 收集整个记录窗口内的 mouseDown、mouseUp 事件与划火柴触发（防止最后一帧后还有）
        remaining_downs = []
        remaining_ups = []
        remaining_triggers = []
        remaining_rate_key_downs = []
        remaining_rate_key_ups = []
        remaining_start_key_events = []
        with self._lock:
            while self._mouse_downs:
                ts, x, y = self._mouse_downs.popleft()
                remaining_downs.append({"ts": ts, "x": x, "y": y})
            while self._mouse_ups:
                ts, x, y = self._mouse_ups.popleft()
                remaining_ups.append({"ts": ts, "x": x, "y": y})
            while self._matchstick_triggers:
                ts, x, y = self._matchstick_triggers.popleft()
                remaining_triggers.append({"ts": ts, "x": x, "y": y})
            while self._rate_key_downs:
                ts, name = self._rate_key_downs.popleft()
                remaining_rate_key_downs.append({"ts": ts, "name": name})
            while self._rate_key_ups:
                ts, name = self._rate_key_ups.popleft()
                remaining_rate_key_ups.append({"ts": ts, "name": name})
            while self._start_key_events:
                ts, name, pressed = self._start_key_events.popleft()
                remaining_start_key_events.append({"ts": ts, "name": name, "pressed": pressed})

        log = {
            "fps": fps,
            "duration_s": duration_s,
            "actual_duration_ms": duration,
            "roi": ROI_B,
            "frames": frames,
            "remaining_mouse_downs": remaining_downs,
            "remaining_mouse_ups": remaining_ups,
            "remaining_matchstick_triggers": remaining_triggers,
            "remaining_rate_key_downs": remaining_rate_key_downs,
            "remaining_rate_key_ups": remaining_rate_key_ups,
            "remaining_start_key_events": remaining_start_key_events,
        }
        with open(out_dir / "log.json", "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False, indent=2)

        # 简单输出时间线
        print("\n[时间线]")
        print(f"{'帧':>4} | {'时间ms':>8} | {'state':>12} | {'费用':>6} | {'左键':>4} | {'down':>4} | {'up':>2} | {'火柴':>4} | {'F↓':>3} | {'F↑':>2} | {'SPC':>3}")
        print("-" * 86)
        for f in frames:
            downs = len(f["mouse_downs_in_frame"])
            ups = len(f["mouse_ups_in_frame"])
            triggers = len(f["matchstick_triggers_in_frame"])
            rate_downs = len(f["rate_key_downs_in_frame"])
            rate_ups = len(f["rate_key_ups_in_frame"])
            start_events = len(f["start_key_events_in_frame"])
            cost_str = f"{f['cost_white']:<6}" if f["cost_white"] is not None else "  -   "
            print(
                f"{f['idx']:>4} | {f['ts_rel_ms']:>8.1f} | {f['state']:>12} | "
                f"{cost_str:>6} | {'D' if f['left_down'] else 'U':>4} | {downs:>4} | {ups:>2} | {triggers:>4} | "
                f"{rate_downs:>3} | {rate_ups:>2} | {start_events:>3}"
            )

        print(f"\n已保存: {out_dir / 'log.json'}")
        print(f"按 {self._start_key.upper()} 再次记录，{self._exit_key.upper()} 退出，{self._matchstick_key.upper()} 划火柴，{self._rate_key.upper()} 倍率键\n")


if __name__ == "__main__":
    args = _parse_args()
    diag = RateTransitionDiagnostic(
        matchstick_delay_ms=args.matchstick_delay,
        matchstick_key=args.matchstick_key,
        rate_key=args.rate_key,
        start_key=args.start_key,
        exit_key=args.exit_key,
        backend=args.backend,
        wgc_interval=args.wgc_interval,
    )
    diag.run(fps=args.fps, duration_s=args.duration)
