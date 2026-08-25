import asyncio
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pynput import mouse, keyboard

import action
import core.base.constants as constants
from core.vision.avatar_matcher import AvatarMatcherBase, create_avatar_matcher
from core.capture.capture import WindowCapture
from core.base.paths import game_template
from core.control.executor import ScriptExecutor
from core.control.retry_handler import StageRetryHandler
from core.control.stage_selector import StageSelector
from core.game_state.cost_bar_sync_cc import CostBarSyncCC
from core.game_state.cost_bar_calibration import list_calibrations
from core.vision.ocr_engine import OCREngine
from core.game_state.region_state_timer import RegionStateTimer
from core.map.tile_pos import TilePosCalculator, load_stage_dimensions
from models.raw_recording import RawRecording, RawAction, Keyframe, KeyframeType
from models.script_schema import ScriptModel, ActionType, OperatorAction


class _ScriptTimerWrapper:
    """在装载脚本执行期间替代 RegionStateTimer，以固定倍率连续推进。

    RegionStateTimer 依赖视觉倍率采样，在每次 pause/resume 后需要若干 tick
    才能重新稳定到 2x，导致计时器滞后于实际游戏时间。本包装器以固定 2x
    倍率根据 wall-clock 推进，只在 executor 调用 pause/resume 时切换状态，
    从而消除滞后。脚本结束后再把底层 RegionStateTimer 对齐到本包装器的
    elapsed 时间，保证用户可以继续录制而不会出现时间跳变。
    """

    def __init__(self, timer: RegionStateTimer, rate: float = constants.FAST2X_RATE, debug: bool = False):
        self._timer = timer
        self._rate = rate
        self._debug = debug
        self._base_elapsed = 0.0
        self._accumulated = 0.0
        self._running = False
        self._paused = True
        self._last_resume_time = 0.0

    def sync_start(self, cost_sync=None):
        """从底层计时器读取当前时间，并用费用条修正初始相位，然后开始固定倍率推进。"""
        self._base_elapsed = self._timer.get_elapsed_ms()
        self._accumulated = 0.0
        self._running = True
        self._paused = False
        self._last_resume_time = time.perf_counter()
        if self._debug:
            self._log(f"同步启动 base={self._base_elapsed:.1f}ms rate={self._rate}x")
        if cost_sync is not None:
            self._calibrate_to_cost_bar(cost_sync)

    def _calibrate_to_cost_bar(self, cost_sync):
        """用费用条当前帧相位修正包装器的初始时间，只修正当前周期内的相位差。

        包装器在启动时的滞后通常远小于一个费用条周期，因此底层计时器的
        cycle_index 基本正确；这里只把相位对齐到费用条当前帧，避免跨周期
        误判导致越修越偏。
        """
        try:
            roi_gray = cost_sync.capture_roi_gray()
            if roi_gray is None:
                if self._debug:
                    self._log("费用条初始修正: 无法截取 ROI")
                return
            count = int(np.sum(roi_gray > cost_sync.threshold))
            elapsed = self.get_elapsed_ms()
            cost_frame = cost_sync.current_frame(count, elapsed)
            cal = cost_sync.get_calibration(elapsed)
            if self._debug:
                target_frame = cost_sync.target_frame_index(elapsed)
                self._log(
                    f"费用条初始修正: count={count}, elapsed={elapsed:.1f}, "
                    f"cal={cal.name}, target_frame={target_frame}, cost_frame={cost_frame}"
                )
            if cost_frame is None:
                return
            frame_duration = cal.frame_duration_ms
            cycle_duration = cal.cycle_duration_ms()
            offset = cost_sync.frame_offset_ms
            adjusted = max(0.0, elapsed - offset)
            cycle_index = int(adjusted / cycle_duration)
            desired_phase = cost_frame * frame_duration
            corrected = cycle_index * cycle_duration + desired_phase + offset
            diff = corrected - elapsed
            if self._debug:
                self._log(
                    f"费用条初始修正计算: cycle_index={cycle_index}, "
                    f"desired_phase={desired_phase:.1f}, corrected={corrected:.1f}, diff={diff:+.1f}"
                )
            if abs(diff) <= constants.COST_BAR_SYNC_MAX_DIFF_MS:
                self.adjust(diff)
                if self._debug:
                    self._log(
                        f"费用条初始修正已应用 {diff:+.1f}ms -> {self.get_elapsed_ms():.1f}ms "
                        f"(frame={cost_frame}, cycle={cycle_index})"
                    )
            elif self._debug:
                self._log(f"费用条初始修正差值 {diff:.1f}ms 超过阈值，跳过")
        except Exception as e:
            if self._debug:
                self._log(f"费用条初始修正异常: {e}")

    def sync_underlying(self):
        """脚本结束后把底层计时器对齐到本包装器的 elapsed 时间。"""
        # 先冻结包装器，避免对齐期间把用户接管/暂停的等待时间也计入
        self.pause()
        target = self.get_elapsed_ms()
        current = self._timer.get_elapsed_ms()
        diff = target - current
        if abs(diff) > 0.5:
            self._timer.adjust(diff)
            if self._debug:
                self._log(f"同步底层计时器 {current:.1f}ms -> {target:.1f}ms ({diff:+.1f}ms)")
        elif self._debug:
            self._log(f"同步底层计时器 无需调整 current={current:.1f}ms target={target:.1f}ms")

        # 重置底层 tick 基准，丢弃脚本期间残留的切换事件，避免接管后时间翻倍
        try:
            self._timer.reset_tick_baseline(paused=True)
            if self._debug:
                self._log("tick 基准已重置")
        except Exception:
            pass

    def get_elapsed_ms(self) -> float:
        if self._running and not self._paused:
            delta_ms = (time.perf_counter() - self._last_resume_time) * 1000.0 * self._rate
            return self._base_elapsed + self._accumulated + delta_ms
        return self._base_elapsed + self._accumulated

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self):
        if not self._paused:
            now = time.perf_counter()
            self._accumulated += (now - self._last_resume_time) * 1000.0 * self._rate
            self._paused = True
            # 同步到底层计时器，避免 matchstick shield 期间错过模拟的空格键
            try:
                self._timer.pause()
            except Exception:
                pass
            if self._debug:
                self._log(f"pause elapsed={self.get_elapsed_ms():.1f}ms")

    def resume(self):
        if self._paused:
            self._paused = False
            self._last_resume_time = time.perf_counter()
            self._running = True
            # 同步到底层计时器，避免 matchstick shield 期间错过模拟的空格键
            try:
                self._timer.resume()
            except Exception:
                pass
            if self._debug:
                self._log(f"resume elapsed={self.get_elapsed_ms():.1f}ms")

    def adjust(self, offset_ms: float):
        self._accumulated += offset_ms
        if self._debug:
            self._log(f"adjust {offset_ms:+.1f}ms -> {self.get_elapsed_ms():.1f}ms")

    def reset(self):
        # 不允许 reset，否则脚本结束后无法保留准确时间用于同步底层计时器
        if self._debug:
            self._log("reset 被忽略")

    def _log(self, message: str):
        print(f"[脚本计时器] {message}")

    def __getattr__(self, name: str):
        return getattr(self._timer, name)


class ActionRecorder:
    """基于全局输入监听的关键帧录制器。

    录制阶段不再依赖预设编队：
      1. 在编队界面截取干员头像/名称关键帧，建立离线识别模板库；
      2. 进入作战后通过状态机记录 DEPLOY/RETREAT/SKILL 等原始操作，
         使用占位符（__click_<ratio>__ / __grid_r_c__）表示目标；
      3. 停止录制后调用 OfflineResolver 离线解析为 ScriptModel。

    时间基准完全委托给 RegionStateTimer。
    """

    # 基于 2560x1600 的固定 ROI（选中干员后视角居中，按钮位置固定）
    # side view 下不同地图的摄像机参数不同，按钮中心会系统性偏移，
    # 因此 x/y 在 __init__ 中根据 tile_calc 的 side view 投影锚点动态计算。
    _RETREAT_W = 170
    _RETREAT_H = 160
    _SKILL_W = 250
    _SKILL_H = 200

    # side view 下不同地图的摄像机参数不同，按钮中心会系统性偏移。
    # 这里用一个“等效世界锚点”模型：假设按钮在地图世界坐标系中对应一个固定点 P，
    # 直接使用 tile_pos 的 side view 透视投影把它投到屏幕上。
    # 锚点由 0-1 / 7-18 / 7-12 / 11-5 / 13-10 / 14-10 / TO-1/4/5/6/9 / TO-EX-5
    # 的标定数据拟合得到。
    _RETREAT_ANCHOR = (-1.396806, 1.056352, -0.166007)   # (px, py, pz)
    _SKILL_ANCHOR = (1.323576, -1.863407, -0.463183)     # (px, py, pz)

    _BASE_W = 2560
    _BASE_H = 1600

    # 部署时名称卡 ROI（基于 2560x1600 的绝对像素 x,y,w,h），显示在画面左上角
    _NAME_CARD_X = 0
    _NAME_CARD_Y = 480
    _NAME_CARD_W = 240
    _NAME_CARD_H = 50

    # 整栏截图 ROI 比例：覆盖费用数字到屏幕底部，确保数量区域（1560~1600）被包含
    # 实际截图时会上移 20px，因此保存到关键帧的 effective top = 1380 - 20 = 1360
    _BAR_CAPTURE_TOP_RATIO = 1380 / 1600
    _BAR_CAPTURE_EFFECTIVE_TOP_RATIO = 1360 / 1600
    _BAR_CAPTURE_HEIGHT_RATIO = 240 / 1600

    # 拖拽判定：mouseUp 与 mouseDown 距离超过此阈值视为拖拽（而非点击）
    # 同时影响 pre-deploy 截图触发时机，避免光标/手指刚离开部署栏就截图
    _DRAG_THRESHOLD = 50
    # 方向选择：二次拖拽距离超过此阈值才判定有方向
    _DIR_THRESHOLD = 20
    # 方向选择二次拖拽的起点必须落在第一次拖拽终点 grid 的此像素范围内，否则视为无方向
    _DIR_START_MAX_DIST = 300
    # 拖拽出部署栏后，延迟多久截取整栏关键帧（秒），给 UI 响应/数量数字刷新留出时间
    _BAR_CAPTURE_DELAY = 0.1
    # 名称卡截图延迟（秒），给左上角名称卡 UI 加载留出时间
    _NAME_CARD_CAPTURE_DELAY = 0.05
    # 状态机超时（秒）
    _TIMEOUT_DEPLOY_DIR = 2.0
    _TIMEOUT_UNIT_SELECT = 2.0

    # side 视角下，部署落点相对 tile 中心偏下一行；
    # 判定前把点沿实际落点方向的反方向移回 tile 中心（按 1600 基准缩放）。
    _SIDE_GRID_OFFSET_MAG_PX = 20

    def __init__(
        self,
        capture: WindowCapture,
        timer: Optional[RegionStateTimer] = None,
        stage_code: str = "",
        debug: bool = False,
        debug_cost_bar: bool = False,
        debug_resolver: bool = False,
        debug_screenshot: bool = False,
        debug_loaded_script: bool = False,
        debug_skill_status: bool = False,
        initial_operator_count: int = 0,
        initial_item_count: int = 0,
        support_count: int = 0,
        pause_key: str = "space",
        takeover_hotkey: str = "F9",
        matchstick_hotkeys: Optional[dict] = None,
        cost_bar_calibration_name: Optional[str] = None,
        ocr: Optional[OCREngine] = None,
        avatar_model_name: str = "resnet18",
        resolver_log_callback: Optional[Callable[[str], None]] = None,
        loaded_script: Optional[ScriptModel] = None,
        loaded_script_path: Optional[str] = None,
        probability_retry_enabled: bool = False,
        challenge_mode: bool = False,
        sand_table: bool = False,
        support_friend_index: Optional[int] = None,
        support_skill: int = 1,
        support_module: int = 1,
    ):
        self.capture = capture
        self.timer = timer
        # debug: 录制器自身状态机/原始操作日志
        # debug_cost_bar: RegionStateTimer 费用条检测日志
        # debug_resolver: OfflineResolver 离线识别日志
        # debug_screenshot: DEPLOY/RETREAT/SKILL 调试截图
        # debug_loaded_script: 装载脚本执行日志
        # debug_skill_status: 技能可点击状态 YOLO 检测日志
        self.debug = debug
        self.debug_cost_bar = debug_cost_bar
        self.debug_resolver = debug_resolver
        self.debug_screenshot = debug_screenshot
        self.debug_loaded_script = debug_loaded_script or debug
        self.debug_skill_status = debug_skill_status
        self._pause_key = pause_key
        self._takeover_hotkey = (takeover_hotkey or "F9").strip().upper()
        self._takeover_key = self._parse_hotkey(self._takeover_hotkey)
        self._matchstick_hotkeys = matchstick_hotkeys
        self._cost_bar_calibration_name = cost_bar_calibration_name
        self.avatar_model_name = avatar_model_name
        self.resolver_log_callback = resolver_log_callback

        self.loaded_script = loaded_script
        self.loaded_script_path = loaded_script_path
        self.probability_retry_enabled = probability_retry_enabled
        self.challenge_mode = challenge_mode
        self.sand_table = sand_table
        self.support_friend_index = support_friend_index
        self.support_skill = support_skill
        self.support_module = support_module
        self._probability_retry_triggered = False
        self._retry_handler: Optional[StageRetryHandler] = None
        self._stage_selector: Optional[StageSelector] = None
        self._script_timer_wrapper: Optional[_ScriptTimerWrapper] = None
        self._loaded_script_error: Optional[str] = None
        self._executed_actions: List[OperatorAction] = []
        self._initial_deployed: Dict[Tuple[int, int], str] = {}
        self._initial_bar_state: Optional[List[Dict]] = None
        self._takeover_requested = False
        self._on_takeover_callback: Optional[Callable[[], None]] = None
        self._on_script_executed_callback: Optional[Callable[[], None]] = None
        self._on_timer_adjusted_callback: Optional[Callable[[], None]] = None

        self.initial_operator_count = max(0, initial_operator_count)
        self.initial_item_count = max(0, initial_item_count)
        self.support_count = max(0, support_count)
        self.stage_code = stage_code
        self.ocr = ocr

        if not stage_code:
            raise ValueError("必须提供关卡代号 stage_code")
        dims = load_stage_dimensions(stage_code)
        if dims is None:
            raise ValueError(
                f"无法在 levels.json 中找到关卡代号 '{stage_code}' 的尺寸信息，"
                f"请确认代号正确或补充 levels.json"
            )
        self.grid_cols, self.grid_rows = dims

        w, h = capture.get_window_size()
        left = capture.monitor.get("left", 0)
        top = capture.monitor.get("top", 0)

        self.tile_calc = TilePosCalculator(
            w, h, self.grid_rows, self.grid_cols,
            stage_code=stage_code,
        )

        self._scale_x = w / self._BASE_W
        self._scale_y = h / self._BASE_H

        # 根据 side view 摄像机参数计算技能/撤退按钮中心，并回退到默认值
        self._retreat_x, self._retreat_y = self._compute_retreat_roi()
        self._skill_x, self._skill_y = self._compute_skill_roi()

        # 录制状态
        self._recording = False
        self._stop_requested = False
        self._wait_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        # 关键帧与会话
        self._session_id = ""
        self._session_dir: Optional[Path] = None
        self._keyframes_dir: Optional[Path] = None
        self._keyframes: Dict[str, Keyframe] = {}
        self._raw_actions: List[RawAction] = []
        self._squad_names: List[str] = []
        self._squad_avatars: Dict[str, np.ndarray] = {}
        self._squad_keyframes_captured = False
        self._unknown_counter = 0
        self._avatar_matcher: Optional[AvatarMatcherBase] = None

        # 异步部署栏截图：mouseDown 时立即启动，不阻塞 pynput 监听器
        self._pending_bar_captures: Dict[str, threading.Event] = {}
        self._bar_capture_seq = 0

        # 状态机
        self._state = "WAITING_FOR_START"
        self._pending: Optional[Dict] = None
        self._mouse_down_pos: Optional[Tuple[int, int]] = None
        self._mouse_down_time: Optional[float] = None
        self._selected_unit_grid: Optional[Tuple[int, int]] = None
        self._pending_skill_grid: Optional[Tuple[int, int]] = None
        self._pending_skill_first_ms: Optional[int] = None
        self._pending_skill_last_ms: Optional[int] = None
        self._timeout_timer: Optional[threading.Timer] = None

        # 场上当前占用格子的追踪（用于选中干员），不记录具体名称
        self._active_grids: set = set()

        # pynput 监听器
        self._mouse_listener: Optional[mouse.Listener] = None
        self._keyboard_listener: Optional[keyboard.Listener] = None

        # 时间基准完全委托给 RegionStateTimer
        if self.timer is None:
            self.timer = RegionStateTimer(
                self.capture,
                pause_key=self._pause_key,
                debug=self.debug_cost_bar,
                matchstick_hotkeys=self._matchstick_hotkeys,
                cost_bar_calibration_name=self._cost_bar_calibration_name,
            )
            self._own_timer = True
        else:
            self._own_timer = False
        self._get_time_ms = self.timer.get_elapsed_ms

        # 概率点自动凸图所需的选关/重试处理器
        if self.loaded_script is not None and self.probability_retry_enabled:
            self._stage_selector = StageSelector(self.capture, self.ocr, debug=self.debug)
            self._init_retry_handler()

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def set_initial_operator_count(self, count: int) -> None:
        """在录制器运行期间动态更新初始干员数量。"""
        with self._lock:
            self.initial_operator_count = max(0, count)
        self._log(f"初始干员数量已更新为 {self.initial_operator_count}")

    def set_initial_item_count(self, count: int) -> None:
        """在录制器运行期间动态更新初始道具数量。"""
        with self._lock:
            self.initial_item_count = max(0, count)
        self._log(f"初始道具数量已更新为 {self.initial_item_count}")

    def set_support_count(self, count: int) -> None:
        """在录制器运行期间动态更新是否借用助战干员（0 或 1）。"""
        with self._lock:
            self.support_count = max(0, count)
        self._log(f"助战干员数量已更新为 {self.support_count}")

    @staticmethod
    def _parse_hotkey(key_str: str):
        """把用户输入的快捷键字符串解析为 pynput 可比较的 Key 对象。"""
        name = (key_str or "").strip().lower()
        if not name:
            return None
        special = getattr(keyboard.Key, name, None)
        if special is not None:
            return special
        if len(name) == 1:
            return keyboard.KeyCode.from_char(name)
        return None

    def start(self):
        if self._recording:
            self._log("start() 被调用但已在录制中")
            return
        self._recording = True
        self._stop_requested = False
        self._reset_session()
        self._raw_actions.clear()
        self._keyframes.clear()
        self._squad_names.clear()
        self._squad_avatars.clear()
        self._squad_keyframes_captured = False
        self._active_grids.clear()
        self._unknown_counter = 0
        self._pending_bar_captures.clear()
        self._bar_capture_seq = 0
        self._executed_actions.clear()
        self._initial_deployed.clear()
        self._initial_bar_state = None
        self._takeover_requested = False
        self._probability_retry_triggered = False
        self._loaded_script_error = None
        self._reset_state()

        self._state = "WAITING_FOR_START"
        self._wait_thread = threading.Thread(target=self._wait_for_timer_start, daemon=True)
        self._wait_thread.start()
        self._log("start() 进入 WAITING_FOR_START，等待 cost 检测...")

        # 键盘监听器始终启动，用于响应 F10 停止及接管快捷键；
        # 鼠标监听器在无预装载脚本时立即启动，有预装载脚本时等脚本结束/接管后再启动。
        self._start_keyboard_listener()
        if self.loaded_script is None:
            self._start_mouse_listener()

    def _start_keyboard_listener(self):
        """启动键盘监听器，用于快捷键（停止录制、手动接管等）。"""
        if self._keyboard_listener is not None:
            return
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
        )
        self._keyboard_listener.start()
        self._log("键盘监听器已启动")

    def _start_mouse_listener(self):
        """启动鼠标监听器，开始记录用户操作。"""
        if self._mouse_listener is not None:
            return
        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_move=self._on_move,
        )
        self._mouse_listener.start()
        self._log("鼠标监听器已启动")

    def _stop_recording_listeners(self):
        """停止鼠标/键盘监听器。"""
        if self._mouse_listener:
            self._mouse_listener.stop()
            self._mouse_listener = None
        if self._keyboard_listener:
            self._keyboard_listener.stop()
            self._keyboard_listener = None

    def stop(self) -> ScriptModel:
        if not self._recording:
            self._log("stop() 被调用但不在录制中")
            return self._resolve_recording()
        self._recording = False
        self._stop_requested = True
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        self._cancel_timeout()
        if self._wait_thread is not None:
            self._wait_thread.join(timeout=0.5)
        if self._own_timer and self.timer is not None:
            self.timer.stop()
        # 等待未完成的异步部署栏截图（最多 2 秒）
        if self._pending_bar_captures:
            self._log(f"等待 {len(self._pending_bar_captures)} 个异步部署栏截图完成...")
            deadline = time.perf_counter() + 2.0
            while self._pending_bar_captures and time.perf_counter() < deadline:
                time.sleep(0.01)
            if self._pending_bar_captures:
                self._log(f"警告: 仍有 {len(self._pending_bar_captures)} 个部署栏截图未完成")

        # 兜底：停止时若仍有预输入的技能，按最后一次有效时间落盘
        if self._pending_skill_grid is not None and self._pending_skill_last_ms is not None:
            self._log(
                f"stop() 兜底落盘技能 grid={self._pending_skill_grid} "
                f"time_ms={self._pending_skill_last_ms}ms"
            )
            self._record_raw_skill(self._pending_skill_grid, self._pending_skill_last_ms)
            self._pending_skill_grid = None
            self._pending_skill_first_ms = None
            self._pending_skill_last_ms = None

        self._log(f"停止录制，共录制 {len(self._raw_actions)} 个原始操作")
        return self._resolve_recording()

    def is_recording(self) -> bool:
        return self._recording

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    def is_squad_capture_done(self) -> bool:
        return getattr(self, "_squad_keyframes_captured", False)

    def get_display_time_ms(self) -> float:
        """返回用于悬浮窗显示的时间（毫秒）。

        装载脚本执行期间返回包装器时间（脚本实际推进时间），
        否则返回底层 RegionStateTimer 时间。
        """
        if self._script_timer_wrapper is not None:
            return self._script_timer_wrapper.get_elapsed_ms()
        if self.timer is not None:
            return self.timer.get_elapsed_ms()
        return 0.0

    # ------------------------------------------------------------------
    # 会话与会话目录
    # ------------------------------------------------------------------
    def _reset_session(self):
        self._session_id = str(int(time.time() * 1000))
        self._session_dir = Path("debug") / "recordings" / self._session_id
        self._keyframes_dir = self._session_dir / "keyframes"
        self._screenshots_dir = self._session_dir / "screenshots"
        self._debug_log_path = self._session_dir / "debug.log"
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._keyframes_dir.mkdir(parents=True, exist_ok=True)
        self._screenshots_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"会话目录: {self._session_dir}")

    def _log(self, message: str):
        """在 debug 模式时同时输出到控制台并写入会话日志文件。"""
        if not self.debug:
            return
        line = f"[录制器] {message}"
        print(line)
        if self._debug_log_path is not None:
            try:
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _log_loaded_script(self, message: str):
        """在装载脚本 debug 模式时输出到控制台并写入会话日志文件。"""
        if not self.debug_loaded_script:
            return
        line = f"[装载脚本] {message}"
        print(line)
        if self._debug_log_path is not None:
            try:
                with self._debug_log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def _log_pending_skill(self, prefix: str):
        """调试：打印当前技能预输入状态。"""
        self._log(
            f"{prefix} pending_skill_grid={self._pending_skill_grid} "
            f"first={self._pending_skill_first_ms} last={self._pending_skill_last_ms}"
        )

    def _save_image(self, image: np.ndarray, filename: str, png_compression: int = 1) -> Path:
        path = self._keyframes_dir / filename
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression])
        if ok:
            path.write_bytes(encoded.tobytes())
        return path

    def _save_debug_screenshot(self, prefix: str, time_ms: int,
                                roi: Optional[Tuple[int, int, int, int]] = None):
        """保存调试用截图到会话 screenshots 目录（仅 debug_screenshot 模式）。

        统一使用 capture()（PrintWindow）截取窗口，再按 ROI 裁剪，支持后台/遮挡窗口。
        保存前丢弃 Alpha 通道，避免 BGRA 在部分看图软件里显示成透明蒙版。
        """
        if not self.debug_screenshot or self._screenshots_dir is None:
            return
        try:
            full = self.capture.capture()
            if len(full.shape) == 3 and full.shape[2] == 4:
                full = cv2.cvtColor(full, cv2.COLOR_BGRA2BGR)
            if roi is not None:
                x, y, w, h = roi
                left = self.capture.monitor.get("left", 0)
                top = self.capture.monitor.get("top", 0)
                wx = x - left
                wy = y - top
                img = full[wy : wy + h, wx : wx + w]
            else:
                img = full
            filename = f"{prefix}_{time_ms:08d}.png"
            path = self._screenshots_dir / filename
            ok, encoded = cv2.imencode(".png", img)
            if ok:
                path.write_bytes(encoded.tobytes())
                self._log(f"已保存调试截图: {path.name} ({img.shape[1]}x{img.shape[0]})")
        except Exception as e:
            self._log(f"调试截图保存失败: {e}")

    # ------------------------------------------------------------------
    # 编队界面关键帧
    # ------------------------------------------------------------------
    def _capture_squad_keyframes(self):
        """在编队界面截取头像与名称关键帧，建立初始模板库。"""
        if self.ocr is None or self.initial_operator_count == 0:
            self._log("未提供 OCR 或初始干员数为 0，跳过编队关键帧")
            return

        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)

        name_ratios = constants.SQUAD_NAME_ROI_RATIOS
        avatar_ratios = constants.SQUAD_AVATAR_ROI_RATIOS
        has_support = self.support_count > 0

        # 构建需要截取的槽位索引：
        # - 无助战时，按顺序取前 initial_operator_count 个普通槽位（跳过右上角助战位 12）
        # - 有助战时，前 (initial_operator_count - support_count) 个为普通干员，
        #   最后 support_count 个固定取右上角助战位 12 开始的槽位
        if has_support:
            regular_count = self.initial_operator_count - self.support_count
            roi_indices = list(range(regular_count)) + list(
                range(12, 12 + self.support_count)
            )
        else:
            roi_indices = [
                i for i in range(self.initial_operator_count) if i != 12
            ]

        captured = 0
        for roi_index in roi_indices:
            if roi_index >= len(name_ratios) or roi_index >= len(avatar_ratios):
                break

            rx, ry, rw, rh = name_ratios[roi_index]
            x = left + int(w * rx)
            y = top + int(h * ry)
            roi_w = int(w * rw)
            roi_h = int(h * rh)
            try:
                name_img = self.capture.capture_roi(x, y, roi_w, roi_h)
                if self.debug:
                    self._log(
                        f"编队槽位 {captured} name_roi capture: shape={name_img.shape}, "
                        f"mean={name_img.mean():.1f}"
                    )
            except Exception as e:
                self._log(f"编队槽位 {captured} name_roi 截图失败: {e}")
                continue
            lines = self.ocr.recognize(name_img, min_confidence=0.5)
            best_name = None
            best_conf = 0.0
            for _bbox, (text, conf) in lines:
                if conf > best_conf:
                    best_conf = conf
                    best_name = text
            name = best_name.strip() if best_name else ""
            if not name:
                name = f"__squad_{captured}__"
                self._log(f"编队槽位 {captured} (ROI={roi_index}) 名称识别失败，使用占位符")
            self._squad_names.append(name)

            name_filename = f"squad_name_{captured:02d}.png"
            name_path = self._save_image(name_img, name_filename)
            name_id = f"squad_name_{captured:02d}"
            self._keyframes[name_id] = Keyframe(
                id=name_id,
                path=str(name_path.relative_to(self._session_dir)),
                type=KeyframeType.SQUAD_NAME,
                time_ms=0,
                bar_index=captured,
            )

            ax, ay, aw, ah = avatar_ratios[roi_index]
            ax = left + int(w * ax)
            ay = top + int(h * ay)
            aw = int(w * aw)
            ah = int(h * ah)
            try:
                avatar = self.capture.capture_roi(ax, ay, aw, ah)
                if self.debug:
                    self._log(
                        f"编队槽位 {captured} avatar_roi capture: shape={avatar.shape}, "
                        f"mean={avatar.mean():.1f}"
                    )
            except Exception as e:
                self._log(f"编队槽位 {captured} avatar_roi 截图失败: {e}")
                continue
            self._squad_avatars[name] = avatar

            avatar_filename = f"squad_avatar_{captured:02d}.png"
            avatar_path = self._save_image(avatar, avatar_filename)
            avatar_id = f"squad_avatar_{captured:02d}"
            self._keyframes[avatar_id] = Keyframe(
                id=avatar_id,
                path=str(avatar_path.relative_to(self._session_dir)),
                type=KeyframeType.SQUAD_AVATAR,
                time_ms=0,
                bar_index=captured,
            )

            slot_label = "助战" if roi_index >= 12 else "普通"
            self._log(f"编队槽位 {captured} (ROI={roi_index}, {slot_label}): {name} (conf={best_conf:.2f})")
            captured += 1

    def _preload_avatar_matcher(self):
        """在编队界面预创建头像匹配器并缓存模板特征，减少录制结束后离线解析等待时间。"""
        if not self._squad_avatars:
            self._log("没有可用的编队头像，跳过匹配器预加载")
            return
        try:
            self._avatar_matcher = create_avatar_matcher(
                prefer_resnet=True,
                use_onnx=True,
                input_size=224,
                model_name=self.avatar_model_name,
            )
            providers = getattr(self._avatar_matcher, "providers", None)
            providers_str = f", providers={providers}" if providers else ""
            t0 = time.perf_counter()
            self._avatar_matcher.set_template_cache(self._squad_avatars)
            self._log(
                f"预加载头像匹配器完成: model={self.avatar_model_name}, "
                f"templates={len(self._squad_avatars)}, "
                f"耗时={(time.perf_counter() - t0) * 1000:.1f}ms{providers_str}"
            )
        except Exception as e:
            self._log(f"预加载头像匹配器失败: {e}")
            self._avatar_matcher = None

    # ------------------------------------------------------------------
    # 解析与输出
    # ------------------------------------------------------------------
    def _resolve_recording(self) -> ScriptModel:
        """序列化 RawRecording 并调用离线解析器生成 ScriptModel。"""
        raw = self._build_raw_recording()
        if self._session_dir is not None:
            raw_path = self._session_dir / "raw_recording.json"
            try:
                raw_path.write_text(raw.model_dump_json(indent=2), encoding="utf-8")
                self._log(f"RawRecording 已保存: {raw_path}")
            except Exception as e:
                self._log(f"保存 RawRecording 失败: {e}")
        from core.recording.resolver import OfflineResolver
        self._log(
            f"开始离线解析: avatar_model={self.avatar_model_name}, "
            f"actions={len(raw.actions)}, keyframes={len(raw.keyframes)}"
        )
        t0 = time.perf_counter()
        script = OfflineResolver(
            raw,
            session_dir=self._session_dir,
            ocr=self.ocr,
            debug=self.debug_resolver,
            avatar_model_name=self.avatar_model_name,
            avatar_matcher=self._avatar_matcher,
            log_callback=self.resolver_log_callback,
            initial_deployed=self._initial_deployed,
            initial_bar_state=self._initial_bar_state,
        ).resolve()
        elapsed = (time.perf_counter() - t0) * 1000
        placeholders = [
            a.operator_name for a in script.actions
            if a.operator_name and a.operator_name.startswith("__")
        ]
        self._log(
            f"离线解析完成: total={elapsed:.1f}ms, model={self.avatar_model_name}, "
            f"operators={len(script.operators)}, items={len(script.items)}, "
            f"summons={len(script.summons)}, actions={len(script.actions)}, "
            f"placeholders={len(set(placeholders))}"
        )

        # 若有预装载脚本已执行的操作，合并到解析结果前面
        if self._executed_actions:
            base = self.loaded_script.model_copy(deep=True) if self.loaded_script else script.model_copy(deep=True)
            merged_actions = list(self._executed_actions)
            # 用户录制操作的时间应接在已执行操作之后
            last_executed_ms = max((a.time_ms for a in merged_actions), default=0)
            for a in script.actions:
                copied = a.model_copy(deep=True)
                copied.time_ms = max(copied.time_ms, last_executed_ms)
                merged_actions.append(copied)
            merged_actions.sort(key=lambda x: x.time_ms)
            base.actions = merged_actions
            base.takeover_boundary_index = len(self._executed_actions)

            # 合并用户录制阶段出现的新干员/道具/召唤物/绑定，避免基础脚本里没有这些单位
            seen_ops = set(base.operators)
            for name in script.operators:
                if name not in seen_ops:
                    base.operators.append(name)
                    seen_ops.add(name)

            seen_items = {it.name for it in base.items}
            for it in script.items:
                if it.name not in seen_items:
                    base.items.append(it.model_copy(deep=True))
                    seen_items.add(it.name)

            seen_sums = {s.name for s in base.summons}
            for s in script.summons:
                if s.name not in seen_sums:
                    base.summons.append(s.model_copy(deep=True))
                    seen_sums.add(s.name)

            seen_bindings = {b.operator_name for b in base.summon_bindings}
            for b in script.summon_bindings:
                if b.operator_name not in seen_bindings:
                    base.summon_bindings.append(b.model_copy(deep=True))
                    seen_bindings.add(b.operator_name)

            return base

        return script

    def _build_raw_recording(self) -> RawRecording:
        hints = {
            "initial_item_count": self.initial_item_count,
            "support_count": self.support_count,
            "squad_names": self._squad_names,
        }
        raw = RawRecording(
            stage_code=self.stage_code,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            session_id=self._session_id,
            initial_operator_count=self.initial_operator_count,
            initial_item_count=self.initial_item_count,
            keyframes=self._keyframes,
            actions=self._raw_actions,
            hints=hints,
        )
        raw.sort_actions()
        return raw

    # ------------------------------------------------------------------
    # 坐标工具
    # ------------------------------------------------------------------
    def _win_xy(self, abs_x: int, abs_y: int) -> Tuple[int, int]:
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return abs_x - left, abs_y - top

    def _focus_game_window(self, click: bool = True):
        """将输入焦点切回游戏窗口。

        用户点击悬浮窗的“手动接管”后，焦点会落在悬浮窗上，导致后续模拟的
        暂停键无法作用于游戏。通过点击游戏窗口中心（或仅刷新窗口矩形），
        确保游戏重新获得焦点，暂停键能被正确接收。
        """
        try:
            monitor = self.capture.monitor
            if monitor is None:
                self.capture.refresh_rect()
                monitor = self.capture.monitor
            center_x = monitor["left"] + monitor["width"] // 2
            center_y = monitor["top"] + monitor["height"] // 2
            if click:
                self._log(f"点击游戏窗口中心以重新聚焦: ({center_x}, {center_y})")
                action.select_at(center_x, center_y)
                time.sleep(0.15)
            else:
                self._log(f"刷新游戏窗口矩形: ({monitor['left']},{monitor['top']} "
                          f"{monitor['width']}x{monitor['height']})")
        except Exception as e:
            self._log(f"聚焦游戏窗口失败: {e}")

    def _nearest_grid(self, win_x: int, win_y: int, side: bool = False) -> Optional[Tuple[int, int]]:
        # side 视角下部署落点相对 tile 中心偏下一行；
        # 判定前把测试点沿实际落点方向的反方向移回 tile 中心。
        if side:
            ox, oy = self.tile_calc.get_side_deploy_offset_vector(
                offset_px_base=self._SIDE_GRID_OFFSET_MAG_PX
            )
            win_x -= int(round(ox))
            win_y -= int(round(oy))
        # side 视角下使用投影四边形命中测试，避免“落在 A 格内但离 B 格中心更近”的误判
        if side:
            hit = self.tile_calc.hit_test(win_x, win_y, side=True)
            if hit is not None:
                return hit
        best = None
        best_dist = float("inf")
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                sx, sy = self.tile_calc.get_screen_pos(r, c, side=side)
                d = (sx - win_x) ** 2 + (sy - win_y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (r, c)
        if best_dist > (150 * min(self._scale_x, self._scale_y)) ** 2:
            return None
        return best

    def _total_bar_slots(self) -> int:
        return self.initial_operator_count + self.initial_item_count

    def _bar_layout_total(self) -> int:
        """返回用于命中检测/坐标计算的部署栏槽位总数。

        实际栏位可能因技能/召唤物临时增加，但 UI 始终保留最多 12 个槽位的
        宽度；返回 max(实际, 12) 确保点击最左侧召唤物/道具位也能被识别。
        """
        return max(self._total_bar_slots(), 12)

    def _bar_positions(self) -> Dict[int, Tuple[int, int]]:
        """返回部署栏索引（0 为最右侧）到绝对屏幕坐标的映射。"""
        total = self._bar_layout_total()
        if total == 0:
            return {}
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        bar_y = int(h * 1500 / 1600)
        cell_w = w / total
        positions = {}
        for i in range(total):
            cx = w - cell_w * (i + 0.5)
            positions[i] = (left + int(cx), top + bar_y)
        return positions

    def _bar_index_at(self, win_x: int, win_y: int) -> Optional[int]:
        total = self._bar_layout_total()
        if total == 0:
            return None
        positions = self._bar_positions()
        cell_w = self.capture.get_window_size()[0] / total
        half = cell_w / 2
        for i, (cx, cy) in positions.items():
            rel_cx = cx - self.capture.monitor.get("left", 0)
            rel_cy = cy - self.capture.monitor.get("top", 0)
            if rel_cx - half <= win_x <= rel_cx + half and rel_cy - half <= win_y <= rel_cy + half:
                self._log(f"_bar_index_at hit idx={i} pos=({rel_cx:.0f},{rel_cy:.0f})")
                return i
        self._log(f"_bar_index_at miss pos=({win_x:.0f},{win_y:.0f})")
        return None

    def _in_deploy_bar_area(self, win_x: int, win_y: int) -> bool:
        """判断鼠标是否仍处于整个部署栏矩形区域内（而非单个 slot 的命中框）。

        截图/取消部署的判定应基于完整区域，避免光标位于 slot 间隙或栏边缘时
        被误判为已移出部署区。这里使用 12 槽位宽度，确保技能生成的召唤物位
        也被包含在内。
        """
        total = self._total_bar_slots()
        if total == 0:
            return False
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)

        layout_total = self._bar_layout_total()
        cell_w = w / layout_total
        bar_top = top + int(h * self._BAR_CAPTURE_TOP_RATIO) - 20
        bar_bottom = top + h
        bar_left = left + max(0, int(w - cell_w * layout_total))
        bar_right = left + w

        return bar_left <= win_x <= bar_right and bar_top <= win_y <= bar_bottom

    def _bar_capture_roi(self) -> Tuple[int, int, int, int]:
        """返回当前窗口下部署栏截图的绝对屏幕 ROI (x, y, w, h)。

        根据实际 UI 轻微上移 20px，确保头像区域完整落入截图。
        """
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        x = left
        y = top + int(h * self._BAR_CAPTURE_TOP_RATIO) - 20
        roi_w = w
        roi_h = int(h * self._BAR_CAPTURE_HEIGHT_RATIO)
        return x, y, roi_w, roi_h

    def _capture_bar_keyframe(self, time_ms: int) -> str:
        """截取整个部署栏关键帧（pre-deploy 状态）。

        部署栏截图对时序敏感，使用 mss 直接截取 ROI（前台屏幕截图）而非 PrintWindow，
        避免 DWM/PrintWindow 缓存导致截到旧帧，同时减少编码耗时。
        """
        x, y, roi_w, roi_h = self._bar_capture_roi()
        try:
            img = self.capture.capture_roi(x, y, roi_w, roi_h)
        except Exception:
            img = np.zeros((roi_h, roi_w, 4), dtype=np.uint8)
        keyframe_id = f"bar_{time_ms:08d}"
        path = self._save_image(img, f"{keyframe_id}.png")
        self._keyframes[keyframe_id] = Keyframe(
            id=keyframe_id,
            path=str(path.relative_to(self._session_dir)),
            type=KeyframeType.DEPLOY_BAR,
            time_ms=time_ms,
            roi=(0.0, self._BAR_CAPTURE_EFFECTIVE_TOP_RATIO, 1.0, self._BAR_CAPTURE_HEIGHT_RATIO),
        )
        self._log(f"已保存整栏关键帧 {keyframe_id} ({roi_w}x{roi_h})")
        return keyframe_id

    def _capture_team_bar_keyframe(self, time_ms: int) -> str:
        """截取初始完整部署区关键帧（TEAM_BAR），用于离线识别所有干员/道具费用与数量。"""
        x, y, roi_w, roi_h = self._bar_capture_roi()
        try:
            img = self.capture.capture_roi(x, y, roi_w, roi_h)
            if self.debug:
                self._log(
                    f"TEAM_BAR capture: roi=({x},{y},{roi_w},{roi_h}), "
                    f"shape={img.shape}, mean={img.mean():.1f}"
                )
        except Exception as e:
            self._log(f"TEAM_BAR capture_roi 失败: {e}，保存全黑占位图")
            img = np.zeros((roi_h, roi_w, 4), dtype=np.uint8)
        keyframe_id = f"team_bar_{time_ms:08d}"
        path = self._save_image(img, f"{keyframe_id}.png")
        self._keyframes[keyframe_id] = Keyframe(
            id=keyframe_id,
            path=str(path.relative_to(self._session_dir)),
            type=KeyframeType.TEAM_BAR,
            time_ms=time_ms,
            roi=(0.0, self._BAR_CAPTURE_EFFECTIVE_TOP_RATIO, 1.0, self._BAR_CAPTURE_HEIGHT_RATIO),
        )
        self._log(f"已保存初始部署区关键帧 {keyframe_id} ({roi_w}x{roi_h})")
        return keyframe_id

    def _capture_deploy_name_card_keyframe(self, time_ms: int) -> Optional[str]:
        """截取部署时左上角的名称卡关键帧，用于离线 OCR 识别道具/召唤物真实名称。"""
        keyframe_id = f"namecard_{time_ms:08d}"
        return self._capture_name_card_keyframe_with_id(keyframe_id, time_ms)

    def _capture_name_card_keyframe_with_id(
        self, keyframe_id: str, time_ms: int
    ) -> Optional[str]:
        """使用预先分配的关键帧 ID 保存名称卡关键帧。"""
        if self._session_dir is None or self._keyframes_dir is None:
            return None
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)

        x = left
        y = top + int(h * self._NAME_CARD_Y / self._BASE_H)
        roi_w = int(w * self._NAME_CARD_W / self._BASE_W)
        roi_h = int(h * self._NAME_CARD_H / self._BASE_H)
        try:
            img = self.capture.capture_roi(x, y, roi_w, roi_h)
        except Exception as e:
            self._log(f"名称卡截图失败: {e}")
            return None
        if len(img.shape) == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

        path = self._save_image(img, f"{keyframe_id}.png")
        self._keyframes[keyframe_id] = Keyframe(
            id=keyframe_id,
            path=str(path.relative_to(self._session_dir)),
            type=KeyframeType.DEPLOY_NAME_CARD,
            time_ms=time_ms,
            roi=(
                self._NAME_CARD_X / self._BASE_W,
                self._NAME_CARD_Y / self._BASE_H,
                self._NAME_CARD_W / self._BASE_W,
                self._NAME_CARD_H / self._BASE_H,
            ),
        )
        self._log(f"已保存名称卡关键帧 {keyframe_id} ({roi_w}x{roi_h})")
        return keyframe_id

    def _schedule_bar_capture(self, delay: float = 0.0) -> str:
        """安排一次异步部署栏截图，返回预先分配的 keyframe_id。

        mouseDown 时 delay=0 立即启动，避免阻塞 pynput 监听器；
        调用方把返回的 keyframe_id 直接写入 RawAction，后台线程完成截图后
        创建对应 Keyframe 对象。
        """
        with self._lock:
            pending_id = f"pending_bar_{self._bar_capture_seq:04d}"
            self._bar_capture_seq += 1
            event = threading.Event()
            self._pending_bar_captures[pending_id] = event

        def worker():
            try:
                if delay > 0:
                    time.sleep(delay)
                capture_time_ms = int(self._get_time_ms())
                self._capture_bar_keyframe_with_id(pending_id, capture_time_ms)
                self._log(f"异步部署栏截图完成 {pending_id} @ {capture_time_ms}ms")
            except Exception as e:
                self._log(f"异步部署栏截图异常 {pending_id}: {e}")
                # 即使失败也保存一张空白关键帧，避免解析器找不到 keyframe
                self._capture_bar_keyframe_with_id(pending_id, int(self._get_time_ms()))
            finally:
                event.set()
                self._pending_bar_captures.pop(pending_id, None)

        threading.Thread(target=worker, daemon=True).start()
        return pending_id

    def _schedule_name_card_capture(self, delay: float = 0.0) -> str:
        """安排一次异步名称卡截图，返回预先分配的 keyframe_id。

        名称卡 UI 需要短暂时间加载，因此支持 delay；后台线程完成截图后
        创建对应 Keyframe 对象。
        """
        with self._lock:
            pending_id = f"pending_namecard_{self._bar_capture_seq:04d}"
            self._bar_capture_seq += 1
            event = threading.Event()
            self._pending_bar_captures[pending_id] = event

        def worker():
            try:
                if delay > 0:
                    time.sleep(delay)
                capture_time_ms = int(self._get_time_ms())
                self._capture_name_card_keyframe_with_id(pending_id, capture_time_ms)
                self._log(f"异步名称卡截图完成 {pending_id} @ {capture_time_ms}ms")
            except Exception as e:
                self._log(f"异步名称卡截图异常 {pending_id}: {e}")
                self._capture_name_card_keyframe_with_id(pending_id, int(self._get_time_ms()))
            finally:
                event.set()
                self._pending_bar_captures.pop(pending_id, None)

        threading.Thread(target=worker, daemon=True).start()
        return pending_id

    def _capture_bar_keyframe_with_id(self, keyframe_id: str, time_ms: int):
        """使用预先分配的 keyframe_id 保存部署栏关键帧。"""
        x, y, roi_w, roi_h = self._bar_capture_roi()
        try:
            img = self.capture.capture_roi(x, y, roi_w, roi_h)
        except Exception:
            img = np.zeros((roi_h, roi_w, 4), dtype=np.uint8)
        path = self._save_image(img, f"{keyframe_id}.png")
        self._keyframes[keyframe_id] = Keyframe(
            id=keyframe_id,
            path=str(path.relative_to(self._session_dir)),
            type=KeyframeType.DEPLOY_BAR,
            time_ms=time_ms,
            roi=(0.0, self._BAR_CAPTURE_EFFECTIVE_TOP_RATIO, 1.0, self._BAR_CAPTURE_HEIGHT_RATIO),
        )
        self._log(f"已保存整栏关键帧 {keyframe_id} ({roi_w}x{roi_h})")

    @staticmethod
    def _project_anchor_to_screen(
        tile_calc, wx: float, wy: float, wz: float
    ) -> Tuple[float, float]:
        """把等效世界锚点通过 tile_calc 的 side view 投影矩阵投到屏幕坐标。"""
        matrix = tile_calc._get_transform_matrix(side=True)
        px, py, _, pw = np.dot(matrix, np.array([wx, wy, wz, 1.0]))
        sx = (1 + px / pw) / 2 * tile_calc.screen_width
        sy = (1 - py / pw) / 2 * tile_calc.screen_height
        return sx, sy

    def _compute_retreat_roi(self) -> Tuple[int, int]:
        """根据 side view 摄像机参数计算撤退按钮 ROI 左上角坐标，失败则回退默认值。"""
        try:
            sx, sy = self._project_anchor_to_screen(self.tile_calc, *self._RETREAT_ANCHOR)
            return int(round(sx - self._RETREAT_W / 2)), int(round(sy - self._RETREAT_H / 2))
        except Exception:
            return 1145, 510

    def _compute_skill_roi(self) -> Tuple[int, int]:
        """根据 side view 摄像机参数计算技能按钮 ROI 左上角坐标，失败则回退默认值。"""
        try:
            sx, sy = self._project_anchor_to_screen(self.tile_calc, *self._SKILL_ANCHOR)
            return int(round(sx - self._SKILL_W / 2)), int(round(sy - self._SKILL_H / 2))
        except Exception:
            return 1615, 885

    def _in_fixed_roi(self, win_x: int, win_y: int, base_x: int, base_y: int,
                      w: int, h: int) -> bool:
        x1 = base_x * self._scale_x
        y1 = base_y * self._scale_y
        x2 = x1 + w * self._scale_x
        y2 = y1 + h * self._scale_y
        return x1 <= win_x <= x2 and y1 <= win_y <= y2

    # ------------------------------------------------------------------
    # 状态机辅助
    # ------------------------------------------------------------------
    def _cancel_timeout(self):
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
            self._timeout_timer = None

    def _set_timeout(self, delay: float, callback):
        self._cancel_timeout()
        self._timeout_timer = threading.Timer(delay, callback)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()

    def _reset_state(self):
        if self._pending_skill_grid is not None:
            self._log(
                f"_reset_state 丢弃未落盘的技能预输入 "
                f"grid={self._pending_skill_grid} last={self._pending_skill_last_ms}ms"
            )
        self._log(f"_reset_state 从 {self._state} 重置为 IDLE")
        with self._lock:
            self._state = "IDLE"
            self._pending = None
            self._selected_unit_grid = None
            self._pending_skill_grid = None
            self._pending_skill_first_ms = None
            self._pending_skill_last_ms = None
            self._cancel_timeout()

    def _wait_for_timer_start(self):
        self._log("_wait_for_timer_start 开始")

        # 等待主窗口完成最小化，避免截图捕捉到 GUI 自身
        time.sleep(2.0)

        if self.loaded_script is not None:
            # 预装载脚本模式：从编队界面自动进入战斗，然后执行脚本
            self._enter_battle_from_squad()
            try:
                self._execute_loaded_script()
            except Exception as e:
                error_msg = f"装载脚本执行异常: {e}"
                import traceback
                tb = traceback.format_exc()
                # 无论是否开启 debug 都输出到控制台，避免静默吞掉错误
                print(f"[录制器] {error_msg}")
                print(tb)
                self._loaded_script_error = f"{error_msg}\n{tb}"
                self._log(error_msg)
                self._log(tb)
                # 尽量把游戏暂停，避免用户在不知情的情况下时间继续推进
                try:
                    action.press_key(self._pause_key)
                    self.timer.pause()
                except Exception:
                    pass
                # 恢复 RegionStateTimer 热键监听
                try:
                    self.timer.reconnect_hotkey()
                except Exception:
                    pass
                # 隐藏接管按钮
                if self._on_takeover_callback is not None:
                    try:
                        self._on_takeover_callback(False)
                    except Exception:
                        pass
                self._script_timer_wrapper = None

            with self._lock:
                self._state = "IDLE"

            # 有预装载脚本且执行完成后才启动鼠标监听器（键盘监听器已在 start() 启动）
            if not self._stop_requested:
                self._start_mouse_listener()
        else:
            # 1. 先截取编队界面关键帧
            try:
                self._capture_squad_keyframes()
                self._squad_keyframes_captured = True
            except Exception as e:
                self._log(f"编队关键帧截取异常: {e}")

            # 2. 预加载头像匹配器并缓存模板特征（在编队界面完成，不占用战斗时间）
            try:
                self._preload_avatar_matcher()
            except Exception as e:
                self._log(f"预加载头像匹配器异常: {e}")

            # 3. 启动计时器并等待费用条开始
            if not self.timer.is_running():
                self._log("启动计时器 (use_cost_detection=True)")
                self.timer.start(use_cost_detection=True)

            while not self._stop_requested:
                info = self.timer.tick()
                if info.get("started"):
                    # 费用条开始计时，立即截取一次完整部署区作为初始 TEAM_BAR 关键帧
                    try:
                        team_bar_id = self._capture_team_bar_keyframe(int(info.get("elapsed_ms", 0)))
                        self._log(f"计时器已启动 (elapsed={info['elapsed_ms']:.1f}ms)，已保存初始部署区 {team_bar_id}")
                    except Exception as e:
                        self._log(f"初始部署区截图异常: {e}")

                    with self._lock:
                        self._state = "IDLE"
                    break
                time.sleep(self.timer.frame_ms / 1000.0)

        # 持续 tick，保持计时器活跃
        while not self._stop_requested:
            self.timer.tick()
            time.sleep(self.timer.frame_ms / 1000.0)

        self._log("_wait_for_timer_start 结束")

    def _enter_battle_from_squad(self):
        """预装载脚本模式下，从编队界面点击确认开始进入战斗。"""
        self._log("预装载脚本模式，自动点击进入战斗")
        try:
            if self._stage_selector is None:
                self._stage_selector = StageSelector(self.capture, self.ocr, debug=self.debug)
            asyncio.run(
                self._stage_selector.enter_stage(
                    self.stage_code,
                    direct_start=True,
                    should_stop=lambda: self._stop_requested,
                )
            )
        except Exception as e:
            self._log(f"自动进入战斗异常: {e}")

    def _execute_loaded_script(self):
        """执行预装载脚本；支持概率点失败后自动重试。

        使用固定 2x 的 _ScriptTimerWrapper 代替 RegionStateTimer 作为执行器
        计时器，避免视觉倍率检测滞后导致动作时间漂移。脚本结束后再把包装器
        时间同步回底层 RegionStateTimer，保证用户可以继续录制。
        """
        if self.loaded_script is None:
            return

        self._log_loaded_script("开始执行预装载脚本...")
        with self._lock:
            self._state = "EXECUTING_LOADED_SCRIPT"

        # 通知 UI 显示接管按钮
        if self._on_takeover_callback is not None:
            self._on_takeover_callback(True)

        # 脚本执行期间由包装器直接同步底层计时器状态；
        # 先注销 RegionStateTimer 自己的键盘监听器，避免模拟空格键与监听器竞争
        try:
            self.timer._unregister_hotkey()
        except Exception as e:
            self._log_loaded_script(f"注销计时器键盘监听器异常: {e}")

        # 复制脚本，移除开头 time_ms==0 的倍率切换动作，避免后续强制 2x 后重复切换
        script_to_run = self.loaded_script.model_copy(deep=True)
        script_to_run.actions = [
            a for a in script_to_run.actions
            if not (a.time_ms == 0 and a.action in (ActionType.SPEED_UP, ActionType.SPEED_DOWN))
        ]

        attempt = 0
        while not self._stop_requested and not self._takeover_requested:
            attempt += 1
            self._log_loaded_script(f"第 {attempt} 次执行装载脚本...")

            # 重置计时器，确保每局从 0 开始
            self.timer.reset()
            self.timer.start(use_cost_detection=True)

            # 等待费用条启动
            started = False
            while not self._stop_requested:
                info = self.timer.tick()
                if info.get("started"):
                    started = True
                    break
                time.sleep(self.timer.frame_ms / 1000.0)
            if not started:
                self._loaded_script_error = "等待费用条启动超时"
                break

            # 暂停游戏，准备执行脚本
            action.press_key(self._pause_key)
            deadline = time.perf_counter() + 1.0
            while time.perf_counter() < deadline and not self.timer.is_manual_paused():
                time.sleep(0.01)

            # 单次执行装载脚本
            need_retry = self._execute_loaded_script_once(script_to_run)

            if not need_retry:
                # 脚本执行完毕或被接管，退出重试循环
                break

            # 概率点检查失败，需要重试
            if not self.probability_retry_enabled:
                self._log_loaded_script("概率点检查失败但自动重试未启用，停止执行")
                break

            self._log_loaded_script("准备重新进入关卡并再次执行...")
            ok = asyncio.run(self._retry_probability_checkpoint())
            if not ok:
                self._loaded_script_error = "概率点重试进入关卡失败"
                break
            # 继续下一轮循环

        # 脚本已执行完毕、被接管或重试失败，进入过渡状态：同步场上/部署栏状态、
        # 导出当前部署区的实际名称与数量给离线解析器，给用户明确的等待提示。
        with self._lock:
            self._state = "TRANSITIONING_TO_TAKEOVER"
        self._log("正在转为接管状态，同步部署区与场上状态...")

        # 暂停游戏，切换到用户录制
        # 先点击游戏窗口中心重新聚焦，否则悬浮窗可能抢走焦点导致暂停键失效
        self._focus_game_window()
        action.press_key(self._pause_key)
        # 监听器已注销，直接手动暂停底层计时器
        self.timer.pause()
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and not self.timer.is_manual_paused():
            time.sleep(0.01)

        # 记录已执行的操作（用于最终合并）：非接管时认为脚本已完整执行
        if not self._takeover_requested:
            self._executed_actions = [
                a.model_copy(deep=True) for a in script_to_run.actions
            ]
        else:
            # 接管时只保留已经执行完的部分：使用包装器时间更准确地判断
            wrapper = self._script_timer_wrapper
            if wrapper is not None:
                current_ms = wrapper.get_elapsed_ms()
            else:
                current_ms = self.timer.get_elapsed_ms()
            self._executed_actions = [
                a.model_copy(deep=True)
                for a in script_to_run.actions
                if a.time_ms <= current_ms
            ]

        # 脚本结束，切换回 RegionStateTimer 作为时间源
        self._script_timer_wrapper = None

        # 通知 UI 恢复录制按钮
        if self._on_takeover_callback is not None:
            self._on_takeover_callback(False)

        # 恢复 RegionStateTimer 的键盘监听器，以便用户继续录制时暂停键生效
        try:
            self.timer.reconnect_hotkey()
        except Exception as e:
            self._log(f"重连计时器键盘监听器异常: {e}")

        self._log(
            f"预装载脚本执行结束，已执行 {len(self._executed_actions)} 个操作，"
            f"接管={self._takeover_requested}"
        )

    def _execute_loaded_script_once(self, script_to_run: ScriptModel) -> bool:
        """单次执行装载脚本。

        调用前要求游戏已暂停、RegionStateTimer 已启动。
        返回 True 表示概率点检查失败，需要重新进入关卡重试。
        """
        # 创建 ScriptExecutor，使用固定倍率计时器包装器，避免 RegionStateTimer
        # 在 pause/resume 后倍率检测滞后导致的时间漂移。
        wrapper = _ScriptTimerWrapper(
            self.timer, rate=constants.FAST2X_RATE, debug=self.debug_loaded_script
        )
        self._script_timer_wrapper = wrapper
        executor = ScriptExecutor(
            self.capture, self.ocr, action,
            debug=self.debug_loaded_script,
            debug_skill_status=self.debug_skill_status,
        )
        executor.timer = wrapper
        executor.on_timer_adjusted = self._on_timer_adjusted_callback
        executor.on_special_behavior_failed = lambda: self._on_probability_checkpoint_failed(executor)

        # 费用条同步：普通模式使用 10s 前后不同校准表，合约 tag 使用单表
        if self._cost_bar_calibration_name:
            cost_sync = CostBarSyncCC(
                self.capture,
                calibration_name=self._cost_bar_calibration_name,
                debug=self.debug_cost_bar,
            )
        else:
            cost_sync = CostBarSyncCC(
                self.capture,
                calibration_name="normal",
                calibration_schedule=[
                    (0.0, "normal_early"),
                    (10000.0, "normal"),
                ],
                debug=self.debug_cost_bar,
            )
        executor.set_cost_sync(cost_sync)
        self._log_loaded_script(
            f"费用条同步初始化: calibration_name={self._cost_bar_calibration_name or 'normal'} "
            f"schedule={getattr(cost_sync, '_schedule', None)}"
        )

        # 先加载原始脚本，用于初始校准；同步助战标记，保证部署栏排序与点击位置正确
        borrow_support = self.support_count > 0
        executor.load_script(script_to_run, borrow_support=borrow_support)
        # 录制器的 RegionStateTimer 已经按游戏倍率返回缩放后的游戏时间，
        # 不需要 executor 再用 _speed2x_ref_ms 做转换。
        executor.set_timer_returns_game_time(True)
        # 保留 executor 内部的暂停后费用条重同步：包装器虽然消除了倍率检测滞后，
        # 但仍会被 pause/resume 的按键延迟带偏，需要费用条把每个动作后拉回来。

        # ---- 二倍速启动 ----
        # 进入本函数时游戏已暂停，稍作稳定后直接读取当前时间。
        # 装载脚本模式下计时器已通过费用条启动检测启动，不再用 calibrate_timer_at_pause
        # 做帧级校准，避免费用条周期歧义导致初始漂移。
        self._log_loaded_script("游戏已暂停，读取计时器当前点...")
        time.sleep(1.0)

        current_ms = self.timer.get_elapsed_ms()
        self._log_loaded_script(f"RegionStateTimer 当前点 current_ms={current_ms:.1f}")

        # RegionStateTimer 已按游戏倍率缩放时间，因此不需要像 main.py 那样压缩动作时间；
        # 只需把游戏切到二倍速，executor 直接按原始 time_ms 等待即可。
        action.press_key(action.speed_key())
        time.sleep(0.5)

        self._log_loaded_script("恢复游戏，开始执行脚本...")
        action.press_key(self._pause_key)
        # 监听器已注销，直接手动恢复底层计时器
        self.timer.resume()
        deadline = time.perf_counter() + 1.0
        while time.perf_counter() < deadline and self.timer.is_manual_paused():
            time.sleep(0.01)
        time.sleep(0.1)

        self._log_loaded_script(
            f"恢复后 RegionStateTimer elapsed={self.timer.get_elapsed_ms():.1f} "
            f"manual_paused={self.timer.is_manual_paused()}"
        )

        # 与包装器同步启动时间点，此后 executor.wait_until 使用固定 2x 推进
        wrapper.sync_start(cost_sync=cost_sync)
        self._log_loaded_script(
            f"wrapper 启动后 elapsed={wrapper.get_elapsed_ms():.1f}"
        )

        # 在独立事件循环中执行，同时轮询接管请求
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run_executor_with_takeover(executor))
        finally:
            loop.close()
            # 关闭后把当前线程的事件循环置空，避免后续代码拿到已关闭的 loop
            asyncio.set_event_loop(None)
            # 脚本结束后把底层 RegionStateTimer 对齐回包装器时间，以便继续录制
            self._log_loaded_script(
                f"脚本执行结束/接管，wrapper elapsed={wrapper.get_elapsed_ms():.1f}"
            )
            wrapper.sync_underlying()
            self._log_loaded_script(
                f"同步回 RegionStateTimer 后 elapsed={self.timer.get_elapsed_ms():.1f}"
            )

        # 同步脚本执行后的场上部署状态到录制器，
        # 这样用户后续进行撤退/技能操作时才能正确选中已部署单位。
        pool = getattr(executor, "pool", None)
        if pool is not None:
            self._active_grids.update(pool._deployed.values())
            self._initial_deployed = {
                grid: name for name, grid in pool._deployed.items()
            }
            self._log(
                f"同步脚本执行后的场上状态: 已部署 {len(self._initial_deployed)} 个单位"
            )

            # 同步部署栏状态：执行器内部已跟踪道具/召唤物/干员的剩余与排序，
            # 直接用它推断当前部署区槽位数量，并把实际名称导出给解析器。
            op_slots, item_slots = pool.get_bar_slot_counts()
            self.initial_operator_count = op_slots
            self.initial_item_count = item_slots
            self._initial_bar_state = pool.export_initial_bar_state()
            self._log(
                f"同步脚本执行后的部署栏状态: "
                f"operator_like_slots={op_slots}, item_slots={item_slots}, "
                f"slots={[s['name'] for s in self._initial_bar_state]}"
            )

        # 返回是否需要概率点重试
        if self._takeover_requested or self._stop_requested:
            return False
        if self._probability_retry_triggered:
            self._probability_retry_triggered = False
            return True
        return False

    async def _run_executor_with_takeover(self, executor: ScriptExecutor):
        """在 executor 运行的同时轮询接管/停止请求。"""
        exec_task = asyncio.create_task(executor.run())
        while not exec_task.done():
            if self._takeover_requested or self._stop_requested:
                executor._stop_event.set()
                exec_task.cancel()
                try:
                    await exec_task
                except asyncio.CancelledError:
                    pass
                break
            await asyncio.sleep(0.05)

    def take_over(self):
        """用户点击手动接管：标记接管请求，由执行循环检测后暂停并切换模式。"""
        if self._state != "EXECUTING_LOADED_SCRIPT":
            return
        self._takeover_requested = True
        self._log("收到手动接管请求")

    def set_takeover_callback(self, callback: Optional[Callable[[bool], None]]):
        """设置接管模式切换回调，参数 True 表示进入接管模式（显示接管按钮）。"""
        self._on_takeover_callback = callback

    def set_timer_adjusted_callback(self, callback: Optional[Callable[[], None]]):
        """设置计时器主动调整后的回调，用于即时刷新悬浮窗显示。"""
        self._on_timer_adjusted_callback = callback

    # ------------------------------------------------------------------
    # 概率点自动凸图
    # ------------------------------------------------------------------
    def _init_retry_handler(self):
        """初始化关卡重试处理器，用于概率点失败后重新进入关卡。"""
        try:
            template_path = str(game_template("loss.png"))
            failed_template_path = str(game_template("failed.png"))
            mission_end_template_path = str(game_template("retry.png"))
            self._retry_handler = StageRetryHandler(
                self.capture,
                self._stage_selector,
                template_path=template_path,
                failed_template_path=failed_template_path,
                mission_end_template_path=mission_end_template_path,
                debug=self.debug,
            )
            self._log("概率点重试处理器已初始化")
        except Exception as e:
            self._log(f"概率点重试处理器初始化失败: {e}")
            self._retry_handler = None

    def _on_probability_checkpoint_failed(self, executor: ScriptExecutor):
        """概率点检查失败回调：标记需要重试并停止当前执行器。"""
        if not self.probability_retry_enabled:
            return
        self._probability_retry_triggered = True
        executor._stop_event.set()
        self._log_loaded_script("概率点检查失败，将重新进入关卡并再次执行脚本")

    async def _retry_probability_checkpoint(self) -> bool:
        """执行概率点失败后的重试：退出当前关卡并重新进入。"""
        if self._retry_handler is None:
            self._log("概率点重试处理器未初始化，无法重试")
            return False
        self._log("开始概率点重试流程：退出关卡并重新进入...")
        try:
            ok = await self._retry_handler.handle_leak_once(
                self.stage_code,
                should_stop=lambda: self._stop_requested,
                challenge_mode=self.challenge_mode,
                sand_table=self.sand_table,
                borrow_support=self.support_count > 0,
                support_friend_index=self.support_friend_index,
                support_skill=self.support_skill,
                support_module=self.support_module,
            )
            if ok:
                self._log("概率点重试：重新进入关卡成功")
            else:
                self._log("概率点重试：重新进入关卡失败")
            return ok
        except Exception as e:
            self._log(f"概率点重试异常: {e}")
            return False

    def _now_ms(self) -> float:
        t = self._get_time_ms()
        self._log(f"_now_ms={t:.1f}ms")
        return t

    # ------------------------------------------------------------------
    # 操作记录
    # ------------------------------------------------------------------
    def _record_raw_deploy(
        self,
        click_ratio_from_right: float,
        grid: Tuple[int, int],
        direction: Optional[str],
        time_ms: int,
        keyframe_ids: List[str],
    ):
        """记录 DEPLOY 原始动作。

        使用点击位置相对窗口右边缘的比例（0=最右，1=最左），而非基于初始总数的
        bar_index，避免战斗中 slot 总数变化后几何映射出错。
        名称卡关键帧已在 _start_deploy_drag 时捕获，直接通过 keyframe_ids 传入。
        """
        action_obj = RawAction(
            time_ms=time_ms,
            action=ActionType.DEPLOY,
            target_ref=f"__click_{click_ratio_from_right:.6f}__",
            grid=grid,
            direction=direction,
            keyframe_ids=keyframe_ids,
        )
        with self._lock:
            self._raw_actions.append(action_obj)
            self._active_grids.add(grid)
        self._save_debug_screenshot("deploy", time_ms)
        self._log(
            f"RAW DEPLOY click_ratio={click_ratio_from_right:.4f} -> {grid} "
            f"dir={direction} time_ms={time_ms} keyframes={keyframe_ids}"
        )

    def _record_raw_retreat(self, grid: Tuple[int, int], time_ms: int):
        action_obj = RawAction(
            time_ms=time_ms,
            action=ActionType.RETREAT,
            target_ref=f"__grid_{grid[0]}_{grid[1]}__",
            grid=grid,
            keyframe_ids=[],
        )
        with self._lock:
            self._raw_actions.append(action_obj)
            self._active_grids.discard(grid)
        self._save_debug_screenshot("retreat", time_ms)
        self._log(f"RAW RETREAT {grid} time_ms={time_ms}")

    def _record_raw_skill(self, grid: Tuple[int, int], time_ms: int):
        action_obj = RawAction(
            time_ms=time_ms,
            action=ActionType.SKILL,
            target_ref=f"__grid_{grid[0]}_{grid[1]}__",
            grid=grid,
        )
        with self._lock:
            self._raw_actions.append(action_obj)
        self._save_debug_screenshot("skill", time_ms)
        self._log(f"RAW SKILL {grid} time_ms={time_ms}")

    # ------------------------------------------------------------------
    # 输入回调
    # ------------------------------------------------------------------
    def _start_deploy_drag(self, win_x: int, win_y: int, bar_idx: int) -> None:
        """从部署栏开始一次新的拖拽部署，state 必须先由调用方置为 DRAGGING 或已重置。"""
        time_ms = int(self._now_ms())
        w, _ = self.capture.get_window_size()
        # 记录点击位置相对窗口右边缘的比例，避免后续 slot 总数变化导致映射偏差
        click_ratio_from_right = (w - win_x) / w
        with self._lock:
            self._state = "DRAGGING"
            self._pending = {
                "type": "DEPLOY",
                # 保留 bar_index 仅用于调试显示
                "bar_index": bar_idx,
                "click_ratio_from_right": click_ratio_from_right,
                "start_pos": (win_x, win_y),
                "time_ms": time_ms,
                "keyframe_id": None,
                "name_card_keyframe_id": None,
                "bar_captured": False,
            }
        self._log(
            f"mouseDown 点击位置=({win_x:.0f},{win_y:.0f}) "
            f"ratio_from_right={click_ratio_from_right:.4f} "
            f"(slot[{bar_idx}] 仅供参考)，等待移出部署区后截图"
        )

    def _on_click(self, abs_x, abs_y, button, pressed):
        if not self._recording:
            return
        win_x, win_y = self._win_xy(abs_x, abs_y)

        with self._lock:
            state = self._state

        if state == "WAITING_FOR_START":
            return

        self._log(f"mouse{'Down' if pressed else 'Up'} state={state} pos=({win_x:.0f},{win_y:.0f})")

        if pressed:
            self._mouse_down_pos = (win_x, win_y)
            self._mouse_down_time = time.perf_counter()

            if state == "IDLE":
                # 部署栏 slot 在 mouseDown 时由点击位置确定，后续不再修改；
                # 截图仅由 _on_move 在鼠标移出部署区后触发。
                bar_idx = self._bar_index_at(win_x, win_y)
                if bar_idx is not None:
                    self._start_deploy_drag(win_x, win_y, bar_idx)

            elif state == "UNIT_SELECTED":
                # 选中干员后，如果用户又回到部署栏点干员，说明要取消当前选中并部署新干员。
                # 这里在 mouseDown 就切换，避免 mouseUp 时被当作“点击空地”丢弃。
                # 切换前若存在技能预输入，先把最后一次有效技能时间落盘。
                bar_idx = self._bar_index_at(win_x, win_y)
                if bar_idx is not None:
                    if self._pending_skill_grid is not None:
                        self._log(
                            f"UNIT_SELECTED 时在部署区 mouseDown，"
                            f"先落盘技能 grid={self._pending_skill_grid} "
                            f"time_ms={self._pending_skill_last_ms}ms"
                        )
                        self._record_raw_skill(self._pending_skill_grid, self._pending_skill_last_ms)
                        self._pending_skill_grid = None
                        self._pending_skill_first_ms = None
                        self._pending_skill_last_ms = None
                    self._log(
                        f"UNIT_SELECTED 时在部署区 mouseDown (slot[{bar_idx}])，"
                        f"取消当前选中并开始新部署"
                    )
                    self._cancel_timeout()
                    self._reset_state()
                    self._start_deploy_drag(win_x, win_y, bar_idx)

            elif state == "AWAITING_DIRECTION":
                bar_idx = self._bar_index_at(win_x, win_y)
                if bar_idx is not None:
                    # 用户在等待方向时又回到部署栏点干员，说明想放弃当前部署并重新部署。
                    # 取消当前方向选择的超时，不记录这次未完成的 DEPLOY，直接开始新拖拽。
                    self._log(
                        f"AWAITING_DIRECTION 时在部署区 mouseDown (slot[{bar_idx}])，"
                        f"取消当前方向选择并开始新部署"
                    )
                    self._cancel_timeout()
                    self._reset_state()
                    self._start_deploy_drag(win_x, win_y, bar_idx)
                else:
                    grid = self._pending.get("grid")
                    dir_invalid = False
                    if grid is not None:
                        gx, gy = self.tile_calc.get_screen_pos(*grid, side=True)
                        max_dist = self._DIR_START_MAX_DIST * min(self._scale_x, self._scale_y)
                        dist = ((win_x - gx) ** 2 + (win_y - gy) ** 2) ** 0.5
                        if dist > max_dist:
                            dir_invalid = True
                            self._log(
                                f" AWAITING_DIRECTION mouseDown 起点距离 grid={grid} 过远 "
                                f"({dist:.0f}px > {max_dist:.0f}px)，视为无方向"
                            )
                    with self._lock:
                        self._pending["dir_down_pos"] = (win_x, win_y)
                        self._pending["dir_invalid"] = dir_invalid
                    self._log(f" AWAITING_DIRECTION mouseDown dir_down=({win_x:.0f},{win_y:.0f})")

        else:
            if state == "DRAGGING":
                if self._pending is None:
                    self._reset_state()
                    return
                start_pos = self._pending.get("start_pos")
                # 快速拖拽可能未触发 on_move，在 mouseUp 处兜底确认
                drag_detected = self._pending.get("drag_detected", False)
                is_dragging_now = start_pos is not None and self._is_dragging(start_pos, (win_x, win_y))
                if not drag_detected and not is_dragging_now:
                    self._log("未检测到有效拖拽，取消")
                    self._reset_state()
                    return
                self._log(f"mouseUp 兜底确认有效拖拽 slot[{self._pending['bar_index']}]")

                if self._in_deploy_bar_area(win_x, win_y):
                    self._log("拖拽释放位置仍在部署栏，取消")
                    self._reset_state()
                    return
                grid = self._nearest_grid(win_x, win_y, side=True)
                self._log(f" DRAGGING mouseUp nearest_grid={grid} pos=({win_x:.0f},{win_y:.0f})")
                if grid is not None:
                    # 兜底：快速拖拽可能未触发 on_move，在确认有效部署前补一张
                    if not self._pending.get("bar_captured", False):
                        keyframe_id = self._schedule_bar_capture(delay=self._BAR_CAPTURE_DELAY)
                        name_card_id = self._schedule_name_card_capture(delay=self._NAME_CARD_CAPTURE_DELAY)
                        self._pending["keyframe_id"] = keyframe_id
                        self._pending["name_card_keyframe_id"] = name_card_id
                        self._pending["bar_captured"] = True
                        self._log(
                            f"mouseUp 兜底启动 pre-deploy 截图 {keyframe_id} "
                            f"name_card={name_card_id}"
                        )
                    with self._lock:
                        self._state = "AWAITING_DIRECTION"
                        self._pending["grid"] = grid
                    self._set_timeout(self._TIMEOUT_DEPLOY_DIR, self._on_deploy_timeout)
                    self._log(f"等待方向选择 @ {grid}")
                else:
                    self._log("拖拽未落在场上，取消")
                    self._reset_state()

            elif state == "AWAITING_DIRECTION":
                self._cancel_timeout()
                if self._pending is None:
                    self._reset_state()
                    return
                dir_pos = self._pending.get("dir_down_pos")
                grid = self._pending["grid"]
                click_ratio_from_right = self._pending["click_ratio_from_right"]
                time_ms = self._pending["time_ms"]
                direction = None
                if dir_pos is not None and not self._pending.get("dir_invalid", False):
                    dx = win_x - dir_pos[0]
                    dy = win_y - dir_pos[1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist > self._DIR_THRESHOLD:
                        if abs(dx) > abs(dy):
                            direction = "right" if dx > 0 else "left"
                        else:
                            direction = "down" if dy > 0 else "up"
                self._log(
                    f" AWAITING_DIRECTION mouseUp "
                    f"ratio={click_ratio_from_right:.4f} grid={grid} dir={direction}"
                )
                # 先取出关键帧 ID，再重置状态机
                keyframe_id = self._pending.get("keyframe_id")
                name_card_id = self._pending.get("name_card_keyframe_id")
                self._reset_state()
                keyframe_ids = [k for k in (keyframe_id, name_card_id) if k]
                self._record_raw_deploy(click_ratio_from_right, grid, direction, time_ms, keyframe_ids)

            elif state == "UNIT_SELECTED":
                in_retreat = self._in_fixed_roi(win_x, win_y,
                                      self._retreat_x, self._retreat_y,
                                      self._RETREAT_W, self._RETREAT_H)
                in_skill = self._in_fixed_roi(win_x, win_y,
                                      self._skill_x, self._skill_y,
                                      self._SKILL_W, self._SKILL_H)
                self._log(
                    f" UNIT_SELECTED mouseUp in_retreat={in_retreat} in_skill={in_skill} "
                    f"pos=({win_x:.0f},{win_y:.0f})"
                )
                self._log_pending_skill("UNIT_SELECTED入口")
                if in_retreat:
                    # 撤退优先，清空可能存在的技能预输入
                    self._log(" 命中撤退ROI，清空技能预输入并记录RETREAT")
                    self._pending_skill_grid = None
                    self._pending_skill_first_ms = None
                    self._pending_skill_last_ms = None
                    self._record_raw_retreat(self._selected_unit_grid, int(self._now_ms()))
                    self._reset_state()
                    return
                if in_skill:
                    # 已有技能预输入时，若落点命中另一名已部署干员，视为 normal 视角下误选新干员，
                    # 用最后一次有效技能时间落盘并切换选中；否则继续累积技能预输入时间。
                    hit_grid = self._field_unit_hit(win_x, win_y)
                    self._log(f" 技能ROI内 hit_grid={hit_grid} selected={self._selected_unit_grid}")
                    if (
                        self._pending_skill_grid is not None
                        and hit_grid is not None
                        and hit_grid != self._selected_unit_grid
                    ):
                        self._log(
                            f" 技能ROI内命中其他干员 {hit_grid}，"
                            f"使用最后一次有效技能时间 {self._pending_skill_last_ms}ms"
                        )
                        self._record_raw_skill(self._pending_skill_grid, self._pending_skill_last_ms)
                        self._pending_skill_grid = None
                        self._pending_skill_first_ms = None
                        self._pending_skill_last_ms = None
                        with self._lock:
                            self._state = "UNIT_SELECTED"
                            self._selected_unit_grid = hit_grid
                        self._log(f"选中单位 @ {hit_grid} (技能ROI内命中)")
                        return
                    now_ms = int(self._now_ms())
                    if self._pending_skill_grid is None:
                        self._pending_skill_grid = self._selected_unit_grid
                        self._pending_skill_first_ms = now_ms
                    self._pending_skill_last_ms = now_ms
                    self._log(f" 技能ROI内点击，更新预输入时间 {now_ms}ms")
                    self._log_pending_skill("技能ROI点击后")
                    return

                # 技能/撤退 ROI 外：有预输入技能则落盘，再视落点决定是否选中新干员
                hit_grid = self._field_unit_hit(win_x, win_y)
                self._log(f" ROI外 hit_grid={hit_grid} selected={self._selected_unit_grid}")
                if self._pending_skill_grid is not None:
                    self._log(
                        f" ROI外点击，落盘技能 grid={self._pending_skill_grid} "
                        f"time_ms={self._pending_skill_last_ms}ms"
                    )
                    self._record_raw_skill(self._pending_skill_grid, self._pending_skill_last_ms)
                    self._pending_skill_grid = None
                    self._pending_skill_first_ms = None
                    self._pending_skill_last_ms = None
                else:
                    self._log("UNIT_SELECTED 点击空地/其他，丢弃")

                if hit_grid is not None and hit_grid != self._selected_unit_grid:
                    with self._lock:
                        self._state = "UNIT_SELECTED"
                        self._selected_unit_grid = hit_grid
                    self._log(f"选中单位 @ {hit_grid} (ROI外命中)")
                else:
                    self._reset_state()

            elif state == "IDLE":
                hit = self._field_unit_hit(win_x, win_y)
                if hit is not None:
                    grid = hit
                    with self._lock:
                        self._state = "UNIT_SELECTED"
                        self._selected_unit_grid = grid
                    self._log(f"选中单位 @ {grid} (区域命中)")
                else:
                    self._log(f" IDLE mouseUp 无部署单位 pos=({win_x:.0f},{win_y:.0f})")

    def _on_move(self, abs_x, abs_y):
        if not self._recording:
            return
        win_x, win_y = self._win_xy(abs_x, abs_y)
        with self._lock:
            state = self._state
            pending = self._pending
        if state != "DRAGGING" or pending is None:
            return
        start_pos = pending.get("start_pos")
        if start_pos is None:
            return
        if not self._is_dragging(start_pos, (win_x, win_y)):
            return

        with self._lock:
            self._pending["drag_detected"] = True
            if not self._pending.get("bar_captured", False):
                # 截图在鼠标移出整个部署栏区域后触发；bar_index 已在 mouseDown 时锁定，不受此位置影响
                if not self._in_deploy_bar_area(win_x, win_y):
                    keyframe_id = self._schedule_bar_capture(delay=self._BAR_CAPTURE_DELAY)
                    name_card_id = self._schedule_name_card_capture(delay=self._NAME_CARD_CAPTURE_DELAY)
                    self._pending["keyframe_id"] = keyframe_id
                    self._pending["name_card_keyframe_id"] = name_card_id
                    self._pending["bar_captured"] = True
                    self._log(
                        f"鼠标移出部署区，已启动 pre-deploy 截图 {keyframe_id} "
                        f"name_card={name_card_id} (bar_delay={self._BAR_CAPTURE_DELAY}s, "
                        f"namecard_delay={self._NAME_CARD_CAPTURE_DELAY}s)"
                    )

    def _is_dragging(self, start_pos: Tuple[int, int], current_pos: Tuple[int, int]) -> bool:
        """判断当前位移是否超过拖拽阈值。"""
        dx = current_pos[0] - start_pos[0]
        dy = current_pos[1] - start_pos[1]
        return (dx ** 2 + dy ** 2) ** 0.5 > self._DRAG_THRESHOLD

    def _on_press(self, key):
        if not self._recording:
            return

        with self._lock:
            state = self._state

        # 停止录制快捷键（始终响应）
        if key == keyboard.Key.f10:
            self._log("F10 停止录制")
            self._stop_requested = True
            return

        # 手动接管快捷键（装载脚本执行期间可用）
        if self._takeover_key is not None and key == self._takeover_key:
            if state == "EXECUTING_LOADED_SCRIPT":
                self._log(f"{self._takeover_hotkey} 手动接管")
                self.take_over()
            return

        if state == "WAITING_FOR_START" or state == "EXECUTING_LOADED_SCRIPT":
            return

        char = getattr(key, "char", None)
        self._log(f"key_press char={char} key={key} state={state}")

        if state == "UNIT_SELECTED":
            if self._selected_unit_grid is None:
                return
            if char and char.lower() == action.retreat_key():
                self._log(f" {action.retreat_key().upper()}键撤退 {self._selected_unit_grid}")
                self._cancel_timeout()
                self._record_raw_retreat(self._selected_unit_grid, int(self._now_ms()))
                self._reset_state()
            elif char and char.lower() == action.skill_key():
                now_ms = int(self._now_ms())
                if self._pending_skill_grid is None:
                    self._pending_skill_grid = self._selected_unit_grid
                    self._pending_skill_first_ms = now_ms
                self._pending_skill_last_ms = now_ms
                self._log(f" {action.skill_key().upper()}键技能预输入 {self._selected_unit_grid} time_ms={now_ms}")
                self._log_pending_skill("按E后")

    # ------------------------------------------------------------------
    # 超时回调
    # ------------------------------------------------------------------
    def _on_deploy_timeout(self):
        with self._lock:
            if self._state != "AWAITING_DIRECTION":
                self._log(f"方向选择超时回调被忽略，当前 state={self._state}")
                return
            pending = self._pending
        if pending and pending.get("type") == "DEPLOY":
            click_ratio = pending["click_ratio_from_right"]
            self._log(
                f"方向选择超时 ratio={click_ratio:.4f} grid={pending['grid']}"
            )
            grid = pending["grid"]
            time_ms = pending["time_ms"]
            keyframe_id = pending.get("keyframe_id")
            name_card_id = pending.get("name_card_keyframe_id")
            self._reset_state()
            keyframe_ids = [k for k in (keyframe_id, name_card_id) if k]
            self._record_raw_deploy(
                click_ratio, grid, None,
                time_ms, keyframe_ids,
            )
        else:
            self._reset_state()
        self._log("方向选择超时，已重置状态")

    # ------------------------------------------------------------------
    # 场上单位命中
    # ------------------------------------------------------------------
    def _field_unit_hit(self, win_x: int, win_y: int) -> Optional[Tuple[int, int]]:
        if not self._active_grids:
            return None
        p00 = self.tile_calc.get_screen_pos(0, 0)
        p01 = self.tile_calc.get_screen_pos(0, 1)
        p10 = self.tile_calc.get_screen_pos(1, 0)
        dx = ((p01[0] - p00[0]) ** 2 + (p01[1] - p00[1]) ** 2) ** 0.5
        dy = ((p10[0] - p00[0]) ** 2 + (p10[1] - p00[1]) ** 2) ** 0.5
        radius = min(dx, dy)

        best_grid = None
        best_dist = float("inf")
        with self._lock:
            for grid in self._active_grids:
                sx, sy = self.tile_calc.get_screen_pos(*grid)
                dist = ((sx - win_x) ** 2 + (sy - win_y) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_grid = grid

        if best_grid is not None and best_dist <= radius:
            return best_grid
        return None
