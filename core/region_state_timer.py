import threading
import time
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from pynput.keyboard import Listener

from core.capture import WindowCapture
from core.cost_bar_start import CostBarStartDetector
from core.cost_bar_sync import CostBarSync
from core.cost_bar_sync_cc import CostBarSyncCC
import core.constants as constants


CostBarSyncType = Union[CostBarSync, CostBarSyncCC]


class _RateTemplateMatcher:
    """基于 1x/2x/0.2x 图标模板 + 帧间差分判定倍率状态。"""

    STATE_FAST = "fast"
    STATE_FAST2X = "fast2x"
    STATE_SLOW = "slow"
    STATE_TRANSITION = "transition"

    def __init__(
        self,
        fast_path: str,
        slow_path: str,
        fast2x_path: Optional[str] = None,
        match_confidence: float = constants.RATE_TEMPLATE_MATCH_CONFIDENCE,
        transition_confidence: float = constants.RATE_TEMPLATE_TRANSITION_CONFIDENCE,
        diff_threshold: float = constants.RATE_TEMPLATE_DIFF_THRESHOLD,
        debug: bool = False,
    ):
        self._debug = debug
        self._match_confidence = match_confidence
        self._transition_confidence = transition_confidence
        self._diff_threshold = diff_threshold
        self._mask_threshold = constants.RATE_TEMPLATE_MASK_THRESHOLD
        self._tmpl_fast, self._mask_fast = self._load_template(fast_path)
        self._tmpl_slow, self._mask_slow = self._load_template(slow_path)
        self._tmpl_fast2x, self._mask_fast2x = (
            self._load_template(fast2x_path) if fast2x_path else (None, None)
        )
        self.available = (
            self._tmpl_fast is not None
            and self._tmpl_slow is not None
            and self._mask_fast is not None
            and self._mask_slow is not None
        )
        self._prev_gray: Optional[np.ndarray] = None

    def _load_template(self, path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None
            if img.ndim == 3 and img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                alpha = img[:, :, 3]
                # 掩膜仅保留不透明且高亮的像素，避免把背景/暂停暗化层纳入匹配
                mask = (
                    (alpha > self._mask_threshold)
                    & (gray > self._mask_threshold)
                ).astype(np.uint8) * 255
            elif img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, self._mask_threshold, 255, cv2.THRESH_BINARY)
            else:
                gray = img
                _, mask = cv2.threshold(gray, self._mask_threshold, 255, cv2.THRESH_BINARY)
            return gray, mask
        except Exception:
            return None, None

    def _match_score(self, roi: np.ndarray, tmpl: np.ndarray, mask: np.ndarray) -> float:
        if (
            tmpl is None
            or mask is None
            or roi.shape[0] < tmpl.shape[0]
            or roi.shape[1] < tmpl.shape[1]
        ):
            return 0.0
        try:
            result = cv2.matchTemplate(roi, tmpl, cv2.TM_CCORR_NORMED, mask=mask)
        except Exception:
            return 0.0
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    def _frame_diff(self, gray: np.ndarray) -> float:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return 0.0
        diff = cv2.absdiff(gray, self._prev_gray)
        return float(np.mean(diff))

    def _decide_state(
        self,
        score_fast: float,
        score_slow: float,
        score_fast2x: float,
        mean_diff: float,
    ) -> str:
        """按三个模板相似度决定倍率；2x 需要达到置信度阈值才被采纳。"""
        if (
            score_fast2x >= self._match_confidence
            and score_fast2x >= score_fast
            and score_fast2x >= score_slow
        ):
            return self.STATE_FAST2X
        return self.STATE_FAST if score_fast >= score_slow else self.STATE_SLOW

    def match(self, gray_b: np.ndarray) -> Tuple[float, float, float, float, str]:
        """返回 (score_fast, score_slow, score_fast2x, mean_diff, state)。"""
        score_fast = (
            self._match_score(gray_b, self._tmpl_fast, self._mask_fast)
            if self._tmpl_fast is not None
            else 0.0
        )
        score_slow = (
            self._match_score(gray_b, self._tmpl_slow, self._mask_slow)
            if self._tmpl_slow is not None
            else 0.0
        )
        score_fast2x = (
            self._match_score(gray_b, self._tmpl_fast2x, self._mask_fast2x)
            if self._tmpl_fast2x is not None
            else 0.0
        )
        mean_diff = self._frame_diff(gray_b)
        self._prev_gray = gray_b.copy()
        state = self._decide_state(score_fast, score_slow, score_fast2x, mean_diff)
        return score_fast, score_slow, score_fast2x, mean_diff, state


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
    DEFAULT_ROI_A = constants.DEFAULT_ROI_A
    DEFAULT_ROI_B = constants.DEFAULT_ROI_B

    def __init__(
        self,
        capture: WindowCapture,
        pause_key: str = "p",
        roi_b: Optional[Tuple[int, int, int, int]] = None,
        threshold: int = constants.REGION_WHITE_THRESHOLD,
        b_fast_threshold: int = constants.REGION_B_FAST_THRESHOLD,
        b_slow_threshold: int = constants.REGION_B_SLOW_THRESHOLD,
        slow_rate: float = constants.SLOW_RATE,
        fast2x_rate: float = constants.FAST2X_RATE,
        frame_ms: float = constants.FRAME_MS,
        startup_offset_ms: float = constants.STARTUP_OFFSET_MS,
        slow_to_fast_compensation_frames: float = constants.SLOW_TO_FAST_COMPENSATION_FRAMES,
        fast_to_slow_compensation_frames: float = constants.FAST_TO_SLOW_COMPENSATION_FRAMES,
        fast_to_fast2x_compensation_frames: float = 0.0,
        fast2x_to_fast_compensation_frames: float = 0.0,
        rate_transition_cooldown_frames: int = constants.RATE_TRANSITION_COOLDOWN_FRAMES,
        sampler_interval_ms: float = 7.0,
        cost_template_path: Optional[str] = None,
        debug: bool = False,
        matchstick_hotkeys: Optional[dict] = None,
        use_template_matching: bool = True,
        rate_template_dir: Optional[str] = None,
        cost_bar_calibration_name: Optional[str] = None,
    ):
        self.capture = capture
        self._pause_key = pause_key
        self.roi_b = roi_b or self.DEFAULT_ROI_B
        self.threshold = threshold
        self.b_fast_threshold = b_fast_threshold
        self.b_slow_threshold = b_slow_threshold
        self.slow_rate = slow_rate
        self.fast2x_rate = fast2x_rate
        self.frame_ms = frame_ms
        self.startup_offset_ms = startup_offset_ms
        self.slow_to_fast_compensation_frames = slow_to_fast_compensation_frames
        self.fast_to_slow_compensation_frames = fast_to_slow_compensation_frames
        self.fast_to_fast2x_compensation_frames = fast_to_fast2x_compensation_frames
        self.fast2x_to_fast_compensation_frames = fast2x_to_fast_compensation_frames
        self.rate_transition_cooldown_frames = rate_transition_cooldown_frames
        self._sampler_interval_ms = sampler_interval_ms
        self.debug = debug

        # 划火柴热键配置：{"select_operator": {"key": "r", "compensation_ms": 2.0}, ...}
        self._matchstick_hotkeys = matchstick_hotkeys or {}
        self._matchstick_ignore_until: Optional[float] = None

        # 倍率模板匹配器
        self._rate_matcher: Optional[_RateTemplateMatcher] = None
        self._last_stable_rate_state: Optional[str] = None
        if use_template_matching:
            tmpl_dir = Path(rate_template_dir) if rate_template_dir else Path(__file__).parent / "resource"
            self._rate_matcher = _RateTemplateMatcher(
                str(tmpl_dir / constants.RATE_TEMPLATE_FAST_NAME),
                str(tmpl_dir / constants.RATE_TEMPLATE_SLOW_NAME),
                fast2x_path=str(tmpl_dir / constants.RATE_TEMPLATE_FAST2X_NAME),
                debug=self.debug,
            )
            if self.debug:
                print(f"[区域计时] 模板匹配可用: {self._rate_matcher.available}")

        # 费用条同步修正（支持普通 / 危机合约 tag）
        self._cost_bar_sync: Optional[CostBarSyncType] = None
        self._cost_bar_maxed = False
        self._cost_bar_last_sync_time: Optional[float] = None
        self._cost_bar_sync_warmed_up = False
        self._cost_bar_sync_corrected_while_paused = False
        if cost_bar_calibration_name:
            self._cost_bar_sync = CostBarSyncCC(
                self.capture,
                calibration_name=cost_bar_calibration_name,
                debug=self.debug,
            )
            if self.debug:
                print(f"[区域计时] 费用条同步使用合约模式: {cost_bar_calibration_name}")
        else:
            self._cost_bar_sync = CostBarSyncCC(
                self.capture,
                calibration_name="normal",
                calibration_schedule=[
                    (0.0, "normal_early"),
                    (10000.0, "normal"),
                ],
                debug=self.debug,
            )
            if self.debug:
                print("[区域计时] 费用条同步使用普通模式校准表（前 10s 区分 29 帧）")

        # 加载费用条 MAX 模板，满费后停止修正
        self._cost_max_template: Optional[np.ndarray] = None
        self._cost_max_mask: Optional[np.ndarray] = None
        cost_max_path = Path(__file__).parent / "resource" / constants.COST_MAX_TEMPLATE_NAME
        self._cost_max_template, self._cost_max_mask = self._load_template_with_mask(str(cost_max_path))
        if self._cost_max_template is None and self.debug:
            print(f"[区域计时] 无法加载费用条 MAX 模板: {cost_max_path}")

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

        # 区域 B 高频采样线程
        self._rate_samples: List[Tuple[float, Optional[int], float, Optional[str]]] = []
        self._sampler_thread: Optional[threading.Thread] = None
        self._sampler_stop_event = threading.Event()

        # 费用条同步线程：避免同步耗时阻塞 tick
        self._cost_sync_thread: Optional[threading.Thread] = None
        self._cost_sync_stop_event = threading.Event()

        # 100ms 防抖，避免误触或快速连按导致重复切换
        self._last_toggle_time: Optional[float] = None

        # 暂停键切换事件队列，元素为 (time.perf_counter(), paused)
        self._toggle_events: List[Tuple[float, bool]] = []
        self._lock = threading.RLock()
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

    @staticmethod
    def _load_template_with_mask(
        path: str,
        mask_threshold: int = constants.RATE_TEMPLATE_MASK_THRESHOLD,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """加载模板并生成前景掩膜（BGRA 按 alpha+亮度，BGR/灰度按亮度阈值）。"""
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if img is None:
                return None, None
            if img.ndim == 3 and img.shape[2] == 4:
                gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                alpha = img[:, :, 3]
                mask = (
                    (alpha > mask_threshold) & (gray > mask_threshold)
                ).astype(np.uint8) * 255
            elif img.ndim == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
            else:
                gray = img
                _, mask = cv2.threshold(gray, mask_threshold, 255, cv2.THRESH_BINARY)
            return gray, mask
        except Exception:
            return None, None

    def _capture_rate_state(self) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """截取区域 B 并返回 (白像素数量, 当前倍率, 状态)。

        当模板匹配可用时优先使用模板匹配；否则回退到白像素阈值逻辑。
        """
        try:
            img = self.capture.capture_roi(*self.roi_b)
            if img.size == 0:
                return None, None, None
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            white = int(np.sum(gray > self.threshold))

            if self._rate_matcher is not None and self._rate_matcher.available:
                score_fast, score_slow, score_fast2x, mean_diff, state = self._rate_matcher.match(gray)
                self._last_stable_rate_state = state
                if state == _RateTemplateMatcher.STATE_FAST:
                    rate = 1.0
                elif state == _RateTemplateMatcher.STATE_FAST2X:
                    rate = self.fast2x_rate
                else:
                    rate = self.slow_rate
                return white, rate, state

            # 回退：白像素阈值（无法区分 1x/2x，统一视为 1x）
            rate: Optional[float] = None
            if white > self.b_fast_threshold:
                rate = 1.0
            elif white < self.b_slow_threshold:
                rate = self.slow_rate
            state = "fast" if rate == 1.0 else "slow" if rate == self.slow_rate else None
            return white, rate, state
        except Exception as e:
            if self.debug:
                print(f"[区域计时] ROI {self.roi_b} 截取失败: {e}")
            return None, None, None

    def _white_count(self, roi: Tuple[int, int, int, int]) -> Optional[int]:
        """旧版兼容：返回 ROI 白像素计数。"""
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

    def _sample_rate_state(self) -> Optional[int]:
        """旧版兼容：仅返回白像素计数。"""
        count_b, _, _ = self._capture_rate_state()
        return count_b

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
        """划火柴热键触发：按当前倍率决定时间补偿，并短暂屏蔽 P 键检测。"""
        now = time.perf_counter()
        _, latest_rate, _ = self._get_latest_sample()
        # 未识别到倍率时默认按子弹时间（0.2x）处理；补偿值基于 0.2x 校准，
        # 其他倍率按相对比例放大。
        effective_rate = latest_rate if latest_rate is not None else self.slow_rate
        rate_factor = effective_rate / self.slow_rate
        adjusted_compensation = compensation_ms * rate_factor
        # 300ms 保护期，期间忽略 P 键（覆盖划火柴操作本身对暂停键的按下）
        with self._lock:
            self._matchstick_ignore_until = now + 0.3
            self._scaled_elapsed_ms += adjusted_compensation
        if self.debug:
            print(
                f"[区域计时] 划火柴 {name} 触发，当前倍率={effective_rate}, "
                f"补偿 +{adjusted_compensation:.1f}ms (基准 {compensation_ms:.1f}ms × {rate_factor:.1f}), "
                f"P 键保护 300ms"
            )

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

    def _start_rate_sampler(self):
        """启动区域 B 高频采样线程。"""
        self._stop_rate_sampler()
        self._sampler_stop_event.clear()
        self._sampler_thread = threading.Thread(
            target=self._rate_sampler_loop, daemon=True
        )
        self._sampler_thread.start()

    def _stop_rate_sampler(self):
        """停止区域 B 高频采样线程。"""
        if self._sampler_thread is not None:
            self._sampler_stop_event.set()
            self._sampler_thread.join(timeout=0.5)
            self._sampler_thread = None

    def _start_cost_sync(self):
        """启动费用条同步线程。"""
        self._stop_cost_sync()
        self._cost_sync_stop_event.clear()
        self._cost_sync_thread = threading.Thread(
            target=self._cost_sync_loop, daemon=True
        )
        self._cost_sync_thread.start()

    def _stop_cost_sync(self):
        """停止费用条同步线程。"""
        if self._cost_sync_thread is not None:
            self._cost_sync_stop_event.set()
            self._cost_sync_thread.join(timeout=0.5)
            self._cost_sync_thread = None

    def _cost_sync_loop(self):
        """周期性执行费用条同步修正，避免阻塞 tick。"""
        while not self._cost_sync_stop_event.is_set():
            try:
                t0 = time.perf_counter()
                self._sync_with_cost_bar()
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sleep_ms = max(0.0, constants.COST_BAR_SYNC_INTERVAL_MS - elapsed_ms)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            except Exception as e:
                if self.debug:
                    print(f"[区域计时] 费用条同步线程异常: {e}")
                time.sleep(constants.COST_BAR_SYNC_INTERVAL_MS / 1000.0)

    def _rate_sampler_loop(self):
        """以 _sampler_interval_ms 为间隔持续采样区域 B 倍率。"""
        while not self._sampler_stop_event.is_set():
            try:
                t0 = time.perf_counter()
                count_b, rate, state = self._capture_rate_state()
                if rate is None:
                    rate = self._rate
                sample = (t0, count_b, rate, state)
                with self._lock:
                    self._rate_samples.append(sample)
                    # 只保留最近 200ms 样本，避免无限增长
                    cutoff = t0 - 0.2
                    self._rate_samples = [
                        s for s in self._rate_samples if s[0] > cutoff
                    ]
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                sleep_ms = max(0.0, self._sampler_interval_ms - elapsed_ms)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            except Exception as e:
                if self.debug:
                    print(f"[区域计时] 倍率采样线程异常: {e}")
                time.sleep(self._sampler_interval_ms / 1000.0)

    def _get_latest_sample(self) -> Tuple[Optional[int], Optional[float], Optional[str]]:
        """返回最近一次的 (count_b, rate, state)。"""
        with self._lock:
            if not self._rate_samples:
                return None, None, None
            _, count_b, rate, state = self._rate_samples[-1]
            return count_b, rate, state

    def _get_average_rate(self, start_time: float, end_time: float) -> Optional[float]:
        """返回 [start_time, end_time] 区间内样本的平均倍率。"""
        with self._lock:
            samples = [
                rate for ts, _, rate, _ in self._rate_samples
                if start_time <= ts <= end_time
            ]
        if not samples:
            return None
        return sum(samples) / len(samples)

    def _match_cost_max(self, roi_gray: np.ndarray) -> float:
        """检测费用条 ROI 是否出现 MAX 字样。"""
        tmpl = self._cost_max_template
        mask = self._cost_max_mask
        if (
            tmpl is None
            or roi_gray.shape[0] < tmpl.shape[0]
            or roi_gray.shape[1] < tmpl.shape[1]
        ):
            return 0.0
        try:
            result = cv2.matchTemplate(
                roi_gray, tmpl, cv2.TM_CCOEFF_NORMED, mask=mask
            )
            _, max_val, _, _ = cv2.minMaxLoc(result)
            return float(max_val)
        except Exception:
            return 0.0

    def _sync_with_cost_bar(self):
        """用费用条帧号修正计时器，防止模板倍率检测的系统性漂移累积。

        支持普通模式和危机合约 tag（通过 CostBarSync / CostBarSyncCC）。
        检测到 MAX 字样后停止修正。
        """
        with self._lock:
            if (
                self._cost_bar_sync is None
                or not self._started
                or self._cost_bar_maxed
            ):
                return
            # 运行期间和暂停期间都持续修正：暂停后游戏真正停下来可能需要几帧，
            # 只修正一次容易把“暂停前一帧”误锁进去，持续修正可在稳定后拉回到正确帧。
            now = time.perf_counter()
            last = self._cost_bar_last_sync_time
            if (
                last is not None
                and (now - last) * 1000.0 < constants.COST_BAR_SYNC_INTERVAL_MS
            ):
                return
            self._cost_bar_last_sync_time = now

        try:
            roi_gray = self._cost_bar_sync.capture_roi_gray()
            if roi_gray is None:
                return

            # 检测 MAX 字样，出现后本次战斗不再修正
            if self._match_cost_max(roi_gray) >= constants.COST_MAX_MATCH_CONFIDENCE:
                with self._lock:
                    self._cost_bar_maxed = True
                if self.debug:
                    print("[区域计时] 费用条 MAX 检测，停止费用条同步修正")
                return

            count = int(np.sum(roi_gray > self._cost_bar_sync.threshold))
            with self._lock:
                elapsed = self._scaled_elapsed_ms
            cost_frame = self._cost_bar_sync.current_frame(count, elapsed)
            if cost_frame is None:
                return
            if not self._cost_bar_sync.is_match(count, cost_frame, elapsed):
                return

            cal = self._cost_bar_sync.get_calibration(elapsed)
            if self.debug:
                print(
                    f"[区域计时] 费用条同步使用校准表: {cal.name}, "
                    f"elapsed={elapsed:.1f}ms"
                )
            # 预热：启动补偿阶段以及费用条还在初始帧 0 时不修正，
            # 等费用条真正开始走动（出现非 0 帧）后再启用同步。
            if not self._cost_bar_sync_warmed_up:
                if elapsed >= self.startup_offset_ms and cost_frame != 0:
                    self._cost_bar_sync_warmed_up = True
                    if self.debug:
                        print("[区域计时] 费用条同步预热完成")
                return

            frame_duration = cal.frame_duration_ms
            cycle_duration = cal.cycle_length * frame_duration
            offset = self._cost_bar_sync.frame_offset_ms
            adjusted = max(0.0, elapsed - offset)
            cycle_index = int(adjusted / cycle_duration)
            desired_phase = cost_frame * frame_duration
            candidates = [
                (cycle_index + i) * cycle_duration + desired_phase + offset
                for i in (-1, 0, 1)
            ]
            corrected = min(candidates, key=lambda t: abs(t - elapsed))
            diff = corrected - elapsed

            if abs(diff) > constants.COST_BAR_SYNC_MAX_DIFF_MS:
                return

            with self._lock:
                # 再次检查状态，避免同步期间被重置/MAX
                if not self._started or self._cost_bar_maxed:
                    return
                self._scaled_elapsed_ms = corrected
            if self.debug:
                print(
                    f"[区域计时] 费用条修正 {diff:+.1f}ms "
                    f"cost_frame={cost_frame} -> {corrected:.1f}ms"
                )
        except Exception as e:
            if self.debug:
                print(f"[区域计时] 费用条同步异常: {e}")

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
        print("[区域计时] 等待初始状态: 区域B为1.0x...")
        start = time.perf_counter()
        while time.perf_counter() - start < timeout:
            count_b, rate, state = self._capture_rate_state()
            if self.debug:
                print(f"[区域计时] 等待中 B={count_b} rate={rate} state={state}")
            if rate is not None and rate == 1.0:
                print(f"[区域计时] 初始状态已满足 B={count_b} state={state}")
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
        self._rate_samples = []
        self._cost_bar_maxed = False
        self._cost_bar_sync_warmed_up = False
        self._cost_bar_sync_corrected_while_paused = False
        self._cost_bar_last_sync_time = None
        self._use_cost_detection = (
            use_cost_detection and self._cost_template is not None
        )
        self._cost_detector = None
        self._register_hotkey()
        self._start_rate_sampler()
        self._start_cost_sync()
        if self._use_cost_detection:
            self._cost_detector = CostBarStartDetector(
                self.capture,
                self._cost_template,
                cost_bar_sync=self._cost_bar_sync,
                debug=self.debug,
            )
            if self.debug:
                print("[区域计时] 启用费用条启动检测")
        elif use_cost_detection and self._cost_template is None and self.debug:
            print("[区域计时] COST 模板未加载，回退到区域B启动检测")

    def stop(self):
        self._running = False
        self._stop_rate_sampler()
        self._stop_cost_sync()
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
            self._started = False
            self._rate = 1.0
            self._prev_rate = 1.0
            self._rate_samples.clear()
            self._rate_transition_cooldown = 0
            self._use_cost_detection = False
            self._cost_bar_maxed = False
            self._cost_bar_sync_warmed_up = False
            self._cost_bar_sync_corrected_while_paused = False
            self._cost_bar_last_sync_time = None
            self._toggle_events = [
                ep for ep in self._toggle_events if ep[0] > time.perf_counter()
            ]
        self._last_tick_time = None

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

        t_start = time.perf_counter()
        try:
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

                count_b, rate, state = self._get_latest_sample()
                info["count_b"] = count_b
                info["state"] = state
                if (
                    rate is not None
                    and rate >= 1.0
                    and not self._paused
                ):
                    self._started = True
                    self._prev_paused = self._paused
                    self._last_tick_time = time.perf_counter()
                    print(f"[区域计时] 启动计时 B={count_b} state={state} rate={rate}")
                info["elapsed_ms"] = self._scaled_elapsed_ms
                return info

            # 运行阶段：用采样线程的最近样本判断离散倍率，用区间平均倍率累加时间
            latest_count_b, latest_rate, latest_state = self._get_latest_sample()
            info["count_b"] = latest_count_b
            info["state"] = latest_state

            # 保存旧倍率并计算本帧目标离散倍率
            self._prev_rate = self._rate
            new_rate = latest_rate if latest_rate is not None else self._rate

            # 用本 tick 区间内的平均倍率累加游戏时间
            now = time.perf_counter()
            avg_rate = self._get_average_rate(self._last_tick_time, now) if self._last_tick_time is not None else None
            if avg_rate is None:
                avg_rate = new_rate

            self._update_time(current_rate=avg_rate)

            # 区域 B 倍率判断（带迟滞，避免阈值附近反复切换导致重复补偿）
            if new_rate != self._rate:
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
                    elif new_rate == self.fast2x_rate and self._rate == 1.0:
                        compensation = self.fast_to_fast2x_compensation_frames * self.frame_ms
                        with self._lock:
                            self._scaled_elapsed_ms += compensation
                        self._rate_transition_cooldown = self.rate_transition_cooldown_frames
                        if self.debug:
                            print(
                                f"[区域计时] 1.0x->2.0x 补偿 +{compensation:.1f}ms "
                                f"({self.fast_to_fast2x_compensation_frames} 帧)"
                            )
                    elif new_rate == 1.0 and self._rate == self.fast2x_rate:
                        compensation = -self.fast2x_to_fast_compensation_frames * self.frame_ms
                        with self._lock:
                            self._scaled_elapsed_ms += compensation
                        self._rate_transition_cooldown = self.rate_transition_cooldown_frames
                        if self.debug:
                            print(
                                f"[区域计时] 2.0x->1.0x 补偿 {compensation:.1f}ms "
                                f"(-{self.fast2x_to_fast_compensation_frames} 帧)"
                            )
                self._rate = new_rate
                if self.debug:
                    print(f"[区域计时] 倍率切换为 {new_rate} B={latest_count_b} state={latest_state}")

            if self._rate_transition_cooldown > 0:
                self._rate_transition_cooldown -= 1

            info["elapsed_ms"] = self._scaled_elapsed_ms
            info["paused"] = self._paused
            info["rate"] = self._rate
            return info
        except Exception as e:
            if self.debug:
                print(f"[区域计时] tick 异常: {e}")
            return {
                "running": True,
                "error": str(e),
                "elapsed_ms": self._scaled_elapsed_ms,
            }
        finally:
            if self.debug:
                dur_ms = (time.perf_counter() - t_start) * 1000.0
                if dur_ms > 50.0:
                    print(f"[区域计时] tick 耗时 {dur_ms:.1f}ms")

    def run_loop(self, stop_check=None):
        """阻塞式运行检测循环，直到 stop() 或 stop_check 返回 True。"""
        self.start(use_cost_detection=True)

        while self._running:
            if stop_check and stop_check():
                break
            self.tick()
            time.sleep(self.frame_ms / 1000.0)

        print(f"[区域计时] 停止，最终计时 {self.get_elapsed_ms():.1f}ms")
