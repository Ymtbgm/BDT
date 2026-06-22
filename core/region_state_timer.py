import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pynput.keyboard import Listener

from core.capture import WindowCapture
from core.cost_bar_start import CostBarStartDetector


class RegionStateTimer:
    """根据键盘暂停键和游戏内倍率区域进行 Scaled 计时。

    pause/run 状态由配置的暂停键键盘事件驱动（脚本模拟或用户物理按键都会
    被全局钩子捕获），不再依赖屏幕区域 A 的模板匹配。

    每次暂停键事件会记录精确时间戳，_update_time 按事件分割区间，从而
    消除截图检测带来的相位抖动和切换中点估计误差。

    区域 B (2175, 34, 128, 119) - 控制计时倍率：
      - 白像素 > 1200：1.0x
      - 白像素 < 1000：0.2x（带迟滞）

    启动条件：区域 B > 1200（费用条检测启用时由费用条检测触发）。
    """

    # 默认 ROI 基于 2560x1600 的绝对屏幕坐标 (x, y, w, h)
    # DEFAULT_ROI_A 保留仅供外部调试工具引用
    DEFAULT_ROI_A = (2375, 53, 112, 88)
    DEFAULT_ROI_B = (2175, 34, 128, 119)

    def __init__(
        self,
        capture: WindowCapture,
        pause_key: str = "p",
        roi_b: Optional[Tuple[int, int, int, int]] = None,
        threshold: int = 200,
        b_fast_threshold: int = 1200,
        b_slow_threshold: int = 1000,
        slow_rate: float = 0.2,
        frame_ms: float = 33.333,
        startup_offset_ms: float = 50.0,
        slow_to_fast_compensation_frames: float = 1.6,
        fast_to_slow_compensation_frames: float = 0.4,
        rate_transition_cooldown_frames: int = 5,
        cost_template_path: Optional[str] = None,
        debug: bool = False,
        matchstick_hotkeys: Optional[dict] = None,
    ):
        self.capture = capture
        self._pause_key = pause_key
        self.roi_b = roi_b or self.DEFAULT_ROI_B
        self.threshold = threshold
        self.b_fast_threshold = b_fast_threshold
        self.b_slow_threshold = b_slow_threshold
        self.slow_rate = slow_rate
        self.frame_ms = frame_ms
        self.startup_offset_ms = startup_offset_ms
        self.slow_to_fast_compensation_frames = slow_to_fast_compensation_frames
        self.fast_to_slow_compensation_frames = fast_to_slow_compensation_frames
        self.rate_transition_cooldown_frames = rate_transition_cooldown_frames
        self.debug = debug

        # 划火柴热键配置：{"select_operator": {"key": "r", "compensation_ms": 2.0}, ...}
        self._matchstick_hotkeys = matchstick_hotkeys or {}
        self._matchstick_ignore_until: Optional[float] = None

        # 加载 COST 模板，用于费用条启动检测
        self._cost_template: Optional[np.ndarray] = None
        if cost_template_path is None:
            cost_template_path = str(CostBarStartDetector.default_template_path())
        self._cost_template = CostBarStartDetector.load_template(cost_template_path)
        if self._cost_template is None and self.debug:
            print(f"[区域计时] 无法加载 COST 模板: {cost_template_path}")

        self._running = False
        self._paused = False
        self._prev_paused = False
        self._started = False
        self._rate = 1.0
        self._prev_rate = 1.0
        self._scaled_elapsed_ms = 0.0
        self._last_tick_time: Optional[float] = None
        self._rate_transition_cooldown = 0
        self._cost_detector: Optional[CostBarStartDetector] = None
        self._use_cost_detection = False

        # 100ms 防抖，避免误触或快速连按导致重复切换
        self._last_toggle_time: Optional[float] = None

        # 暂停键切换事件队列，元素为 (time.perf_counter(), paused)
        self._toggle_events: List[Tuple[float, bool]] = []
        self._lock = threading.Lock()
        self._keyboard_listener: Optional[Listener] = None

    @staticmethod
    def _load_template(path: str) -> Optional[np.ndarray]:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            if img.ndim == 3 and img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            elif img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            return gray
        except Exception:
            return None

    def _white_count(self, roi: Tuple[int, int, int, int]) -> Optional[int]:
        try:
            img = self.capture.capture_roi(*roi)
            if img.size == 0:
                return None
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            return int(np.sum(gray > self.threshold))
        except Exception as e:
            if self.debug:
                print(f"[区域计时] ROI {roi} 截取失败: {e}")
            return None

    def _capture_rate_state(self) -> Optional[int]:
        """截取区域 B 并返回白像素计数，用于判断倍率。"""
        return self._white_count(self.roi_b)

    def _on_pause_key(self):
        """暂停键被按下时切换计时器状态并记录时间戳。"""
        now = time.perf_counter()
        with self._lock:
            # 若处于划火柴操作保护期内，忽略暂停键/ESC（防止划火柴中的模拟按键被误识别）
            if self._matchstick_ignore_until is not None and now < self._matchstick_ignore_until:
                if self.debug:
                    print(f"[区域计时] 暂停键/ESC 在划火柴保护期内被忽略")
                return False
            # 100ms 防抖，避免误触或快速连按导致重复切换
            if self._last_toggle_time is not None and (now - self._last_toggle_time) <= 0.1:
                if self.debug:
                    print(f"[区域计时] 暂停键回调触发但防抖过滤，距上次 {(now - self._last_toggle_time)*1000:.1f}ms")
                return False
            self._paused = not self._paused
            self._toggle_events.append((now, self._paused))
            self._last_toggle_time = now
            state_text = "暂停" if self._paused else "运行"
        if self.debug:
            print(f"[区域计时] 暂停键/ESC 触发 @{now:.6f}，切换为 {state_text}")
        return True

    def _register_hotkey(self):
        self._unregister_hotkey()

        def _on_press(key):
            try:
                key_name = None
                if hasattr(key, 'char') and key.char is not None:
                    key_name = key.char.lower()
                elif hasattr(key, 'name') and key.name is not None:
                    key_name = key.name.lower()

                if key_name in (self._pause_key, "esc"):
                    self._on_pause_key()
                    return

                for name, info in self._matchstick_hotkeys.items():
                    if key_name == info.get("key", "").lower():
                        self._on_matchstick_key(name, info.get("compensation_ms", 0.0))
                        return
            except Exception:
                pass

        try:
            self._keyboard_listener = Listener(on_press=_on_press)
            self._keyboard_listener.start()
        except Exception:
            pass

    def _on_matchstick_key(self, name: str, compensation_ms: float):
        """划火柴热键触发：时间补偿 + 短暂屏蔽 P 键检测。"""
        now = time.perf_counter()
        # 300ms 保护期，期间忽略 P 键（覆盖划火柴操作本身对暂停键的按下）
        with self._lock:
            self._matchstick_ignore_until = now + 0.3
            self._scaled_elapsed_ms += compensation_ms
        if self.debug:
            print(f"[区域计时] 划火柴 {name} 触发，补偿 +{compensation_ms}ms，P 键保护 300ms")

    def adjust(self, offset_ms: float):
        """手动补偿/调整已累积的计时（如脚本子进程需要同步推进时间）。"""
        with self._lock:
            self._scaled_elapsed_ms += offset_ms
        if self.debug:
            print(f"[区域计时] 手动调整 {offset_ms:+.1f}ms，当前时间 {self._scaled_elapsed_ms:.1f}ms")

    def shield_matchstick(self, duration_ms: float = 500.0):
        """外部请求进入划火柴保护期（如脚本子进程即将执行 P+ESC 组合）。"""
        now = time.perf_counter()
        with self._lock:
            self._matchstick_ignore_until = now + duration_ms / 1000.0
        if self.debug:
            print(f"[区域计时] 收到划火柴屏蔽请求，屏蔽 {duration_ms:.0f}ms")

    def update_matchstick_hotkeys(self, matchstick_hotkeys: Optional[dict] = None):
        """运行时动态更新划火柴热键配置（补偿值与屏蔽键位）。"""
        with self._lock:
            self._matchstick_hotkeys = matchstick_hotkeys or {}
        if self.debug:
            print(f"[区域计时] 划火柴热键配置已更新: {self._matchstick_hotkeys}")

    def _unregister_hotkey(self):
        if self._keyboard_listener is not None:
            try:
                self._keyboard_listener.stop()
                self._keyboard_listener.join(timeout=1.0)
            except Exception:
                pass
            self._keyboard_listener = None

    def reconnect_hotkey(self):
        """重新注册键盘热键。

        当外部子进程也注册了全局钩子后，本进程的 pynput 钩子可能因 Windows
        钩子链变动而失效；在子进程结束后调用此方法可恢复键盘监听。
        """
        self._unregister_hotkey()
        self._register_hotkey()

    def _update_time(self, current_rate: Optional[float] = None):
        """根据本周期内的暂停键事件，累加经过的游戏时间。

        事件队列中的每个切换点都会把区间切成若干段，运行段按当前倍率累计，
        暂停段不计时，从而消除固定中点估计带来的累积误差。

        注意：结束时的 paused 状态由本周期内最后处理的事件决定，而不是调用方
        在 tick() 开头捕获的 current_paused。这样可以避免事件在捕获和调用
        _update_time 之间到达导致的 _prev_paused 不同步问题。
        """
        now = time.perf_counter()
        if self._started and self._last_tick_time is not None:
            # 限制极端卡顿（如切出游戏）导致的跳秒
            raw_delta_ms = (now - self._last_tick_time) * 1000.0
            max_delta_ms = self.frame_ms * 5.0
            if raw_delta_ms > max_delta_ms:
                if self.debug:
                    print(f"[区域计时] 单帧延迟 {raw_delta_ms:.1f}ms，Clamp 到 {max_delta_ms:.1f}ms")
                raw_delta_ms = max_delta_ms
                now = self._last_tick_time + raw_delta_ms / 1000.0

            with self._lock:
                start_paused = self._prev_paused
                rate = current_rate if current_rate is not None else self._rate
                # 取出并清理本周期内的事件
                events = [
                    (t, p) for t, p in self._toggle_events
                    if self._last_tick_time <= t <= now
                ]
                self._toggle_events = [
                    ep for ep in self._toggle_events if ep[0] > now
                ]

            # 按事件时间点切分区间
            segments = []
            seg_start = self._last_tick_time
            seg_paused = start_paused
            for toggle_time, new_paused in sorted(events, key=lambda x: x[0]):
                if seg_start <= toggle_time <= now:
                    segments.append((seg_start, toggle_time, seg_paused))
                    seg_start = toggle_time
                    seg_paused = new_paused
            segments.append((seg_start, now, seg_paused))

            # 累加各段游戏时间
            counted_ms = 0.0
            for seg_start, seg_end, seg_paused in segments:
                if seg_paused:
                    continue
                duration_ms = (seg_end - seg_start) * 1000.0
                counted_ms += duration_ms * rate

            with self._lock:
                self._scaled_elapsed_ms += counted_ms
                # 周期结束时的 paused 状态以最后一段为准，避免传入的 current_paused 滞后
                self._prev_paused = seg_paused
        else:
            with self._lock:
                self._prev_paused = self._paused
        self._last_tick_time = now

    def _wait_for_initial_state(self, timeout: float = 30.0, interval: float = 0.01):
        print("[区域计时] 等待初始状态: 区域B>1200...")
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            count_b = self._capture_rate_state()
            if self.debug:
                print(f"[区域计时] 等待中 B={count_b}")
            if count_b is not None and count_b > self.b_fast_threshold:
                print(f"[区域计时] 初始状态已满足 B={count_b}")
                return True
            time.sleep(interval)
        print("[区域计时] 等待初始状态超时")
        return False

    def start(self, use_cost_detection: bool = False):
        self._running = True
        self._paused = False
        self._prev_paused = False
        self._started = False
        self._rate = 1.0
        self._prev_rate = 1.0
        # 使用费用条检测时，由检测成功后的 offset_ms 决定初始时间，
        # 不再额外加入 startup_offset_ms，避免初始帧数偏多。
        self._scaled_elapsed_ms = 0.0 if use_cost_detection else self.startup_offset_ms
        self._last_tick_time = None
        self._rate_transition_cooldown = 0
        self._toggle_events = []
        self._last_toggle_time = None
        self._use_cost_detection = (
            use_cost_detection and self._cost_template is not None
        )
        self._cost_detector = None
        self._register_hotkey()
        if self._use_cost_detection:
            self._cost_detector = CostBarStartDetector(
                self.capture,
                self._cost_template,
                debug=self.debug,
            )
            if self.debug:
                print("[区域计时] 启用费用条启动检测")
        elif use_cost_detection and self._cost_template is None and self.debug:
            print("[区域计时] COST 模板未加载，回退到区域B启动检测")

    def stop(self):
        self._running = False
        self._unregister_hotkey()

    def pause(self):
        now = time.perf_counter()
        with self._lock:
            if not self._paused:
                self._paused = True
                self._toggle_events.append((now, True))

    def resume(self):
        now = time.perf_counter()
        with self._lock:
            if self._paused:
                self._paused = False
                self._toggle_events.append((now, False))

    def manual_pause(self):
        self.pause()

    def manual_resume(self):
        self.resume()

    def toggle_manual_pause(self):
        now = time.perf_counter()
        with self._lock:
            self._paused = not self._paused
            self._toggle_events.append((now, self._paused))

    def is_manual_paused(self) -> bool:
        with self._lock:
            return self._paused

    def reset(self):
        with self._lock:
            self._scaled_elapsed_ms = 0.0
        self._last_tick_time = time.perf_counter() if self._started and not self._paused else None

    def get_elapsed_ms(self) -> float:
        self._update_time()
        return self._scaled_elapsed_ms

    def is_running(self) -> bool:
        return self._running

    def is_started(self) -> bool:
        return self._started

    def tick(self) -> dict:
        """手动调用一帧检测。返回当前状态信息字典。"""
        if not self._running:
            return {"running": False}

        with self._lock:
            current_paused = self._paused

        info = {
            "running": True,
            "started": self._started,
            "paused": current_paused,
            "rate": self._rate,
            "state_a": None,
            "count_a": None,
            "count_b": None,
        }

        if not self._started:
            if self._use_cost_detection and self._cost_detector is not None:
                offset_ms = self._cost_detector.tick()
                if offset_ms is not None and not self._paused:
                    self._started = True
                    self._prev_paused = self._paused
                    self._last_tick_time = time.perf_counter()
                    self._scaled_elapsed_ms = offset_ms
                    if self.debug:
                        print(
                            f"[区域计时] 费用条启动检测完成，开始计时，"
                            f"补偿 {offset_ms:.1f}ms，当前时间 {self._scaled_elapsed_ms:.1f}ms"
                        )
                else:
                    if self.debug:
                        print(f"[区域计时] 费用条启动检测中: {self._cost_detector.state}")
                info["elapsed_ms"] = self._scaled_elapsed_ms
                return info

            count_b = self._capture_rate_state()
            info["count_b"] = count_b
            if count_b is not None and count_b > self.b_fast_threshold and not self._paused:
                self._started = True
                self._prev_paused = self._paused
                self._last_tick_time = time.perf_counter()
                print(f"[区域计时] 启动计时 B={count_b}")
            info["elapsed_ms"] = self._scaled_elapsed_ms
            return info

        count_b = self._capture_rate_state()
        info["count_b"] = count_b

        # 保存旧倍率并计算本帧目标倍率，供 _update_time 在切换区间使用
        self._prev_rate = self._rate
        new_rate = self._rate
        if count_b is not None:
            if count_b > self.b_fast_threshold:
                new_rate = 1.0
            elif count_b < self.b_slow_threshold:
                new_rate = self.slow_rate

        self._update_time(current_rate=new_rate)

        # 区域 B 倍率判断（带迟滞，避免阈值附近反复切换导致重复补偿）
        if new_rate != self._rate:
            # 经验补偿：检测本身存在滞后。
            # 冷却期内允许倍率跟随实际状态，但不再重复补偿，
            # 避免切换后几帧因 UI 渐变反复触发补偿。
            # 仅在当前周期实际处于运行（或即将运行）时补偿。
            # 使用 self._paused 而非捕获时的 current_paused，避免 tick 过程中发生
            # 暂停事件后仍进行补偿。
            if not self._paused and self._rate_transition_cooldown == 0:
                if new_rate == 1.0 and self._rate == self.slow_rate:
                    compensation = self.slow_to_fast_compensation_frames * self.frame_ms
                    with self._lock:
                        self._scaled_elapsed_ms += compensation
                    self._rate_transition_cooldown = self.rate_transition_cooldown_frames
                    if self.debug:
                        print(
                            f"[区域计时] 0.2x->1.0x 补偿 +{compensation:.1f}ms "
                            f"({self.slow_to_fast_compensation_frames} 帧)"
                        )
                elif new_rate == self.slow_rate and self._rate == 1.0:
                    compensation = -self.fast_to_slow_compensation_frames * self.frame_ms
                    with self._lock:
                        self._scaled_elapsed_ms += compensation
                    self._rate_transition_cooldown = self.rate_transition_cooldown_frames
                    if self.debug:
                        print(
                            f"[区域计时] 1.0x->0.2x 补偿 {compensation:.1f}ms "
                            f"(-{self.fast_to_slow_compensation_frames} 帧)"
                        )
            self._rate = new_rate
            if self.debug:
                print(f"[区域计时] 倍率切换为 {new_rate} B={count_b}")

        if self._rate_transition_cooldown > 0:
            self._rate_transition_cooldown -= 1

        info["elapsed_ms"] = self._scaled_elapsed_ms
        info["paused"] = self._paused
        info["rate"] = self._rate
        return info

    def run_loop(self, stop_check=None):
        """阻塞式运行检测循环，直到 stop() 或 stop_check 返回 True。"""
        self.start(use_cost_detection=True)

        while self._running:
            if stop_check and stop_check():
                break
            self.tick()
            time.sleep(self.frame_ms / 1000.0)

        print(f"[区域计时] 停止，最终计时 {self.get_elapsed_ms():.1f}ms")
