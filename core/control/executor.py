import asyncio
from typing import Optional, Dict, List, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel

import core.base.constants as constants
import core.vision.cost_recognition as cost_recognition

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper
from core.game_state.timer import StageTimer
from core.vision.ocr_engine import OCREngine
from core.vision.digit_recognizer import DigitRecognizer
from core.game_state.operator_pool import OperatorPool
from core.game_state.cost_bar_sync import CostBarSync
from core.game_state.cost_bar_sync_cc import CostBarSyncCC
from models.script_schema import ScriptModel, ActionType, OperatorAction, SummonBinding


CostBarSyncType = Union[CostBarSync, CostBarSyncCC]


class ExecutorState(BaseModel):
    is_running: bool = False
    is_paused: bool = False
    current_time_ms: int = 0
    stage_code: Optional[str] = None


class ScriptExecutor:
    def __init__(self, capture: WindowCapture, ocr: OCREngine, action_module, debug: bool = False):
        self.capture = capture
        self.ocr = ocr
        self.action = action_module
        self.debug = debug
        self.timer = StageTimer()
        self.script: Optional[ScriptModel] = None
        self.grid: Optional[GridMapper] = None
        self.pool: Optional[OperatorPool] = None
        self.cost_sync: Optional[CostBarSyncType] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._costs_recognized = False
        self._cost_bar_maxed = False
        self._speed2x_ref_ms: Optional[int] = None
        self._timer_returns_game_time: bool = False
        self._disable_resync_at_pause: bool = False
        self._summon_bindings_map: Dict[str, str] = {}
        self.digit_recognizer: Optional[DigitRecognizer] = None
        self.on_timer_adjusted: Optional[Callable[[], None]] = None

    def set_cost_sync(self, cost_sync: Optional[CostBarSyncType]):
        self.cost_sync = cost_sync
        self._cost_bar_maxed = False
    def set_speed2x_reference(self, ref_ms: Optional[int] = None):
        """设置二倍速时间压缩的参考点，用于费用条同步时反推原始游戏内时间。"""
        self._speed2x_ref_ms = ref_ms

    def set_timer_returns_game_time(self, scaled: bool = True):
        """标记外部计时器已经返回缩放后的游戏内时间，不需要再乘倍率。"""
        self._timer_returns_game_time = scaled

    def set_disable_resync_at_pause(self, disabled: bool = True):
        """禁用二倍速暂停后的费用条重同步，避免与精确计时器包装器冲突。"""
        self._disable_resync_at_pause = disabled

    def _notify_timer_adjusted(self):
        """计时器发生主动调整后，通知外部刷新悬浮窗等显示。"""
        if self.on_timer_adjusted is not None:
            try:
                self.on_timer_adjusted()
            except Exception:
                pass

    def _game_time_ms(self, time_ms: int) -> int:
        """将压缩后的现实时间还原为原始游戏内时间，供费用条帧同步使用。

        如果外部计时器已经返回缩放后的游戏时间，则直接返回。
        """
        if self._timer_returns_game_time:
            return time_ms
        if self._speed2x_ref_ms is not None and time_ms > self._speed2x_ref_ms:
            return self._speed2x_ref_ms + (time_ms - self._speed2x_ref_ms) * 2
        return time_ms

    def _frame_duration_ms(self, time_ms: Optional[int] = None) -> float:
        """返回当前校准表下的一帧时长（ms），用于跳帧后同步计时器。"""
        if self.cost_sync is not None and hasattr(self.cost_sync, "get_calibration"):
            try:
                cal = self.cost_sync.get_calibration(
                    self._game_time_ms(time_ms) if time_ms is not None else self.timer.get_elapsed_ms()
                )
                if cal is not None and cal.frame_duration_ms:
                    return cal.frame_duration_ms
            except Exception:
                pass
        return constants.ADVANCE_FRAME_MS

    def _skip_one_frame_and_measure(self, time_ms: Optional[int] = None) -> float:
        """执行一次 p_and_esc_click 跳帧，并根据费用条白像素反推实际前进的游戏时间（ms）。

        通过前后帧号差计算 delta，可自然处理 29 帧跳到下一秒 0 帧的循环边界。
        """
        if self.cost_sync is None:
            self.action.p_and_esc_click()
            return constants.ADVANCE_FRAME_MS

        timer_ms = self.timer.get_elapsed_ms()
        game_time_ms = self._game_time_ms(timer_ms)
        count0 = self.cost_sync.white_pixel_count()
        frame0 = self.cost_sync.current_frame(count0, game_time_ms) if count0 is not None else None

        self.action.p_and_esc_click()

        count1 = self.cost_sync.white_pixel_count()
        frame1 = self.cost_sync.current_frame(count1, game_time_ms) if count1 is not None else None

        if frame0 is None or frame1 is None:
            return constants.ADVANCE_FRAME_MS

        # 计算实际前进了多少帧（循环意义下），再换算成 ms
        behind = self._frames_behind(frame0, frame1, timer_ms)
        frame_duration = self._frame_duration_ms(time_ms if time_ms is not None else timer_ms)
        delta = round(behind * frame_duration, 3)
        if self.debug:
            print(
                f"[跳帧测量] frame0={frame0} frame1={frame1} 前进={behind}帧 "
                f"frame_duration={frame_duration:.3f}ms delta={delta:.1f}ms"
            )
        return delta

    def _estimate_game_time_at_count(
        self, count: int, compressed_time: int
    ) -> Optional[float]:
        """根据白像素数量和压缩时间，估算当前游戏内时间（用于临时诊断）。"""
        if self.cost_sync is None:
            return None
        game_time = self._game_time_ms(compressed_time)
        cal = self.cost_sync.get_calibration(game_time)
        frame = self.cost_sync.current_frame(count, game_time)
        if frame is None:
            return None
        frame_duration = cal.frame_duration_ms
        cycle_duration = cal.cycle_duration_ms()
        base_time = frame * frame_duration
        k = round((game_time - base_time) / cycle_duration)
        return base_time + k * cycle_duration

    def _resync_timer_at_pause(self, compressed_time: int):
        """在二倍速暂停后，根据费用条把计时器对齐到压缩时间轴。

        游戏收到暂停键到真正暂停之间仍有约 1 帧延迟，期间计时器已停、
        游戏还在二倍速前进，导致计时器比游戏时间落后。这里用费用条反推
        实际游戏时间，再把计时器修正为对应的压缩时间，防止误差累积。
        检测到费用条 MAX（费用已满）后，本局不再重同步。
        """
        if self._disable_resync_at_pause:
            if self.debug:
                print("[二倍速计时器重同步] 已禁用，跳过")
            return
        if self.cost_sync is None:
            return
        if not self._timer_returns_game_time and self._speed2x_ref_ms is None:
            return
        if self._cost_bar_maxed:
            return

        # 费用条已满检测：达到 MAX 后帧号不再循环，重同步会反而引入误差
        if hasattr(self.cost_sync, "is_cost_max"):
            roi_gray = self.cost_sync.capture_roi_gray()
            if roi_gray is not None and self.cost_sync.is_cost_max(roi_gray):
                self._cost_bar_maxed = True
                if self.debug:
                    print("[二倍速计时器重同步] 检测到费用条 MAX，本局停止重同步")
                return
            count = self.cost_sync.white_pixel_count(roi_gray)
        else:
            count = self.cost_sync.white_pixel_count()

        if count is None:
            return
        actual_game_ms = self._estimate_game_time_at_count(count, compressed_time)
        if actual_game_ms is None:
            return
        if self._timer_returns_game_time:
            new_timer_ms = actual_game_ms
        else:
            ref = self._speed2x_ref_ms
            if ref is not None and actual_game_ms > ref:
                new_timer_ms = ref + (actual_game_ms - ref) / 2
            else:
                new_timer_ms = actual_game_ms
        old_timer_ms = self.timer.get_elapsed_ms()
        self.timer.adjust(new_timer_ms - old_timer_ms)
        self._notify_timer_adjusted()
        if self.debug:
            game_time = self._game_time_ms(compressed_time)
            target_frame = self.cost_sync.target_frame_index(game_time)
            current_frame = self.cost_sync.current_frame(count, game_time)
            print(
                f"[二倍速计时器重同步] 压缩目标={compressed_time}ms, "
                f"target_frame={target_frame}, current_frame={current_frame}, "
                f"反推游戏时间={actual_game_ms:.1f}ms, "
                f"旧计时器={old_timer_ms:.1f}ms, 新计时器={new_timer_ms:.1f}ms, "
                f"diff={new_timer_ms - old_timer_ms:+.1f}ms"
            )

    def _resync_timer_to_cost_bar(self, time_ms: int):
        """在暂停/子弹时间状态下，根据费用条当前帧把计时器对齐到实际游戏时间。

        用于帧同步跳帧结束后做最终修正，保证计时器和游戏画面完全一致。
        """
        if self.cost_sync is None:
            return
        if not self._timer_returns_game_time and self._speed2x_ref_ms is None:
            return
        if self._cost_bar_maxed:
            return

        count = self.cost_sync.white_pixel_count()
        if count is None:
            return
        actual_game_ms = self._estimate_game_time_at_count(count, time_ms)
        if actual_game_ms is None:
            return
        if self._timer_returns_game_time:
            new_timer_ms = actual_game_ms
        else:
            ref = self._speed2x_ref_ms
            if ref is not None and actual_game_ms > ref:
                new_timer_ms = ref + (actual_game_ms - ref) / 2
            else:
                new_timer_ms = actual_game_ms

        old_timer_ms = self.timer.get_elapsed_ms()
        diff = new_timer_ms - old_timer_ms
        if abs(diff) > constants.COST_BAR_SYNC_MAX_DIFF_MS:
            if self.debug:
                print(
                    f"[费用条同步-最终对齐] 差值 {diff:.1f}ms 超过阈值，跳过"
                )
            return
        if abs(diff) > 0.5:
            self.timer.adjust(diff)
            self._notify_timer_adjusted()
            print(f"{constants.TIMER_ADJUST_MARKER}:{diff}")
            if self.debug:
                game_time = self._game_time_ms(time_ms)
                target_frame = self.cost_sync.target_frame_index(game_time)
                current_frame = self.cost_sync.current_frame(count, game_time)
                print(
                    f"[费用条同步-最终对齐] target_frame={target_frame}, "
                    f"current_frame={current_frame}, "
                    f"timer {old_timer_ms:.1f}ms -> {new_timer_ms:.1f}ms ({diff:+.1f}ms)"
                )

    def calibrate_timer_at_pause(self) -> int:
        """在暂停状态下根据费用条当前帧校准计时器，返回校准后的游戏时间(ms)。"""
        if self.cost_sync is None:
            rough_time = self.timer.get_elapsed_ms()
            if self.debug:
                print(f"[计时校准-暂停] 无费用条同步，使用计时器时间 {rough_time:.1f}ms")
            return rough_time

        count = self.cost_sync.white_pixel_count()
        if count is None:
            rough_time = self.timer.get_elapsed_ms()
            if self.debug:
                print(f"[计时校准-暂停] 无法获取白像素，使用计时器时间 {rough_time:.1f}ms")
            return rough_time

        rough_time = self.timer.get_elapsed_ms()
        current_frame = self.cost_sync.current_frame(count, rough_time)
        if current_frame is None:
            if self.debug:
                print(f"[计时校准-暂停] 无法估算帧号，使用计时器时间 {rough_time:.1f}ms")
            return rough_time

        cal = self.cost_sync.get_calibration(rough_time)
        frame_duration = cal.frame_duration_ms
        cycle_duration = cal.cycle_duration_ms()
        if cycle_duration <= 0:
            if self.debug:
                print(f"[计时校准-暂停] 周期异常，使用计时器时间 {rough_time:.1f}ms")
            return rough_time

        base_time = current_frame * frame_duration
        k = round((rough_time - base_time) / cycle_duration)
        calibrated_time = int(base_time + k * cycle_duration)
        self.timer.adjust(calibrated_time - rough_time)

        if self.debug:
            print(
                f"[计时校准-暂停] 白像素={count}, 当前帧={current_frame}, "
                f"周期={k}, 粗略时间={rough_time:.1f}ms, 校准后={calibrated_time}ms"
            )
        return calibrated_time

    def load_script(self, script: ScriptModel, borrow_support: bool = False, direct_start: bool = False):
        self.script = script
        self.borrow_support = borrow_support
        self.direct_start = direct_start
        self._speed2x_ref_ms = None
        self._timer_returns_game_time = False
        script.sort_actions()
        w, h = self.capture.get_window_size()
        self.grid = GridMapper(
            w, h, script.grid_rows, script.grid_cols,
            stage_code=script.stage_code,
        )
        support_count = 1 if borrow_support else 0
        self.pool = OperatorPool(
            w, h, script.operators, script.items, script.summons,
            support_count=support_count,
        )
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        self.pool.set_window_offset(left, top)

        # 加载召唤物绑定关系，供 RETREAT 时清理对应召唤物
        self._summon_bindings_map = {
            b.operator_name: b.summon_name
            for b in (script.summon_bindings or [])
        }

    def verify_stage_code(self) -> bool:
        if not self.script or not self.script.stage_code:
            return True
        frame = self.capture.capture()
        found = self.ocr.find_text(frame, self.script.stage_code)
        if found is None:
            return False
        return True

    async def wait_until(self, target_ms: int, check_interval: float = 0.005):
        if self.debug:
            print(
                f"[wait_until] target={target_ms}ms "
                f"start_timer={self.timer.get_elapsed_ms():.1f}ms"
            )
        while self.timer.get_elapsed_ms() < target_ms:
            if self._stop_event.is_set():
                if self.debug:
                    print("[wait_until] 被停止请求中断")
                return False
            # 剩余时间 > 5ms 时用 asyncio.sleep 避免空转；
            # 最后 5ms 自旋等待，消除 sleep 精度抖动（Windows 默认 ~15ms）
            if target_ms - self.timer.get_elapsed_ms() > constants.WAIT_SPIN_THRESHOLD_MS:
                await asyncio.sleep(check_interval)
        if self.debug:
            print(
                f"[wait_until] target={target_ms}ms "
                f"end_timer={self.timer.get_elapsed_ms():.1f}ms"
            )
        return True

    def _get_actual_target(self, action: OperatorAction) -> int:
        """对最左三列的 RETREAT/SKILL 提前触发。"""
        if action.action not in (ActionType.RETREAT, ActionType.SKILL):
            return action.time_ms
        grid = action.grid
        if not grid and action.operator_name and not action.is_object:
            grid = self.pool.get_deployed_grid(action.operator_name)
        if grid and grid[1] in (0, 1, 2):
            if self._timer_returns_game_time:
                advance = constants.LOADED_SCRIPT_LEFT_COLS_ADVANCE_MS
                advance_name = "LOADED_SCRIPT"
            else:
                advance = constants.LEFT_COLS_ADVANCE_MS
                advance_name = "LEFT_COLS"
            actual = max(0, action.time_ms - advance)
            if self.debug:
                print(
                    f"[最左列提前] action={action.action.value} "
                    f"grid={grid} timer_returns_game_time={self._timer_returns_game_time} "
                    f"advance={advance_name}({advance}ms) "
                    f"original={action.time_ms}ms actual={actual}ms"
                )
            return actual
        return action.time_ms

    def _wait_target_ms(self, target_ms: int, action: OperatorAction) -> int:
        """二倍速下将 wait_until 目标提前，抵消暂停键延迟导致的触发偏晚。"""
        if (
            self._speed2x_ref_ms is not None
            and action.action != ActionType.PAUSE
            and target_ms > self._speed2x_ref_ms
        ):
            return max(self._speed2x_ref_ms, target_ms - constants.TWOX_EARLY_TRIGGER_MS)
        if self._timer_returns_game_time and not self._cost_bar_maxed:
            return max(0, target_ms - constants.LOADED_SCRIPT_EARLY_TRIGGER_MS)
        return target_ms

    def _abs_pixel(self, row: int, col: int, side: bool = False):
        x, y = self.grid.grid_to_pixel(row, col, side=side)
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return x + left, y + top

    def _ensure_operator_costs(self):
        """确保已执行过一次部署栏费用识别，在首个动作暂停后调用。"""
        if self._costs_recognized:
            return
        self._costs_recognized = True
        costs, manual_support = self._recognize_operator_costs()
        if manual_support and self.pool is not None:
            self.borrow_support = True
            self.pool.set_support_count(1)
        if costs:
            self.pool.set_operator_costs(costs)
            if self.debug:
                print(f"[部署栏OCR] 首个动作暂停时已设置费用: {costs}")
        elif self.debug:
            print("[部署栏OCR] 首个动作暂停时未识别到费用，使用初始序号排序")

    def _recognize_operator_costs(self) -> Tuple[Dict[str, int], bool]:
        """按干员格子精确裁剪并 OCR 识别费用，返回 (costs, manual_support_detected)。"""
        if not self.script or not self.script.operators:
            return {}, False

        # 懒加载 ONNX 数字分类器，优先用于费用识别（与录制器/解析器保持一致）
        if self.digit_recognizer is None:
            try:
                self.digit_recognizer = DigitRecognizer(use_gpu=True)
                if self.debug:
                    print(f"[DigitRecognizer] 已加载，可用={self.digit_recognizer.available}")
            except Exception as e:
                if self.debug:
                    print(f"[DigitRecognizer] 加载失败，将仅使用 OCR: {e}")
                self.digit_recognizer = None

        item_names = [it.name for it in (self.script.items or [])]
        return cost_recognition.recognize_operator_costs(
            capture=self.capture,
            ocr=self.ocr,
            operators=self.script.operators,
            items=item_names,
            support_count=1 if getattr(self, "borrow_support", False) else 0,
            direct_start=self.direct_start,
            debug=self.debug,
            digit_recognizer=self.digit_recognizer,
        )

    async def _advance_frame_in_bullet_time(self):
        """进入子弹时间后调用 p_and_esc_click 推进一帧，再退出子弹时间。

        游戏实际前进了一帧，因此计时器也同步前进一帧，保证悬浮窗时间与
        游戏画面保持一致。
        """
        pos0 = self.pool.get_bar_index_pos(0)
        if pos0:
            self.action.select_at(pos0[0], pos0[1])
            await asyncio.sleep(1.0)
        self.action.p_and_esc_click()
        delta = self._skip_one_frame_and_measure()
        self.timer.adjust(delta)
        self._notify_timer_adjusted()
        print(f"{constants.TIMER_ADJUST_MARKER}:{delta}")
        if pos0:
            self.action.select_at(pos0[0], pos0[1])
            await asyncio.sleep(1.0)

    async def _sync_to_frame(self, time_ms: int):
        """基于费用条白像素进行帧同步，仅当当前帧落后于目标帧（偏早）时跳帧，最多跳 3 帧。"""
        if self.cost_sync is None:
            return

        game_time_ms = self._game_time_ms(time_ms)
        target_frame = self.cost_sync.target_frame_index(game_time_ms)

        for attempt in range(3):
            count = self.cost_sync.white_pixel_count()
            if count is None:
                return
            current_frame = self.cost_sync.current_frame(count, game_time_ms)
            if current_frame is None:
                return
            behind = self._frames_behind(current_frame, target_frame, time_ms)
            ahead = self._frames_behind(target_frame, current_frame, time_ms)
            if self.debug or self.cost_sync.debug:
                print(
                    f"[费用条同步] 目标帧={target_frame}, 当前帧={current_frame}, "
                    f"白像素={count}, 落后={behind}, 超前={ahead}, 尝试={attempt}, "
                    f"timer={self.timer.get_elapsed_ms():.1f}ms"
                )
            if behind == 0 or ahead == 0:
                self._resync_timer_to_cost_bar(time_ms)
                if self.debug or self.cost_sync.debug:
                    print(
                        f"[费用条同步] 已匹配目标帧，最终对齐后 timer="
                        f"{self.timer.get_elapsed_ms():.1f}ms"
                    )
                return
            if behind <= 3:
                await self._advance_frame_in_bullet_time()
                if self.cost_sync.debug:
                    print("[费用条同步] 跳 1 帧")
                continue
            if ahead <= 3:
                # 当前帧比目标帧稍早（循环意义下 ahead 小），直接对齐计时器
                self._resync_timer_to_cost_bar(time_ms)
                if self.debug or self.cost_sync.debug:
                    print(
                        f"[费用条同步] 当前帧超前 {ahead} 帧，直接对齐 timer="
                        f"{self.timer.get_elapsed_ms():.1f}ms"
                    )
                return
            if self.debug or self.cost_sync.debug:
                print("[费用条同步] 偏晚或落后超过3帧，不跳帧")
            return

    async def _sync_to_frame_after_select(self, time_ms: int):
        """在已选中干员（子弹时间）后进行帧同步，仅偏早时跳帧，最多跳 3 帧。"""
        if self.cost_sync is None:
            return

        game_time_ms = self._game_time_ms(time_ms)
        target_frame = self.cost_sync.target_frame_index(game_time_ms)

        for attempt in range(3):
            count = self.cost_sync.white_pixel_count()
            if count is None:
                return
            current_frame = self.cost_sync.current_frame(count, game_time_ms)
            if current_frame is None:
                return
            behind = self._frames_behind(current_frame, target_frame, time_ms)
            ahead = self._frames_behind(target_frame, current_frame, time_ms)
            if self.debug or self.cost_sync.debug:
                print(
                    f"[费用条同步-选中后] 目标帧={target_frame}, 当前帧={current_frame}, "
                    f"白像素={count}, 落后={behind}, 超前={ahead}, 尝试={attempt}, "
                    f"timer={self.timer.get_elapsed_ms():.1f}ms"
                )
            if behind == 0 or ahead == 0:
                self._resync_timer_to_cost_bar(time_ms)
                if self.debug or self.cost_sync.debug:
                    print(
                        f"[费用条同步-选中后] 已匹配目标帧，最终对齐后 timer="
                        f"{self.timer.get_elapsed_ms():.1f}ms"
                    )
                return
            if behind <= 3:
                delta = self._skip_one_frame_and_measure(time_ms)
                self.timer.adjust(delta)
                self._notify_timer_adjusted()
                print(f"{constants.TIMER_ADJUST_MARKER}:{delta}")
                if self.debug or self.cost_sync.debug:
                    print(
                        f"[费用条同步-选中后] 跳 1 帧，timer="
                        f"{self.timer.get_elapsed_ms():.1f}ms"
                    )
                continue
            if ahead <= 3:
                self._resync_timer_to_cost_bar(time_ms)
                if self.debug or self.cost_sync.debug:
                    print(
                        f"[费用条同步-选中后] 当前帧超前 {ahead} 帧，直接对齐 timer="
                        f"{self.timer.get_elapsed_ms():.1f}ms"
                    )
                return
            if self.debug or self.cost_sync.debug:
                print("[费用条同步-选中后] 偏晚或落后超过3帧，不跳帧")
            return

    async def _execute_action_core(self, action: OperatorAction):
        """仅执行操作逻辑，不处理暂停/恢复外壳。"""
        if self._stop_event.is_set():
            return
        # 首个动作暂停后再识别费用，避免启动时截图过早/不稳定
        self._ensure_operator_costs()
        if action.action == ActionType.DEPLOY:
            if not action.operator_name or not action.grid:
                return
            await self._sync_to_frame(action.time_ms)

            if self._stop_event.is_set():
                return
            to_x, to_y = self._abs_pixel(action.grid[0], action.grid[1], side=True)
            from_pos = self.pool.get_deploy_pos(action.operator_name)
            if from_pos is None:
                raise RuntimeError(f"干员 {action.operator_name} 当前不在部署栏可用列表中")
            from_x, from_y = from_pos
            w, h = self.capture.get_window_size()
            self.action.deploy_at(from_x, from_y, to_x, to_y, direction=action.direction, window_w=w, window_h=h)
            self.pool.deploy(action.operator_name, action.grid)
            # 部署涉及鼠标拖拽，同一 cluster 中若紧接着下一操作可能抢占鼠标，稍等 0.5s
            await asyncio.sleep(0.5)

        elif action.action == ActionType.RETREAT:
            # 解析目标格子，并识别特殊目标（干员名存在但无部署记录）
            grid = action.grid
            recorded_grid = None
            if action.operator_name and not action.is_object:
                recorded_grid = self.pool.get_deployed_grid(action.operator_name)
            if not grid:
                grid = recorded_grid
            if not grid:
                raise RuntimeError(f"撤退操作缺少目标格子（干员/道具: {action.operator_name}）")
            is_special = action.operator_name is not None and recorded_grid is None and not action.is_object

            is_left_three_cols = grid[1] in (0, 1, 2)
            if is_left_three_cols:
                advance = constants.LOADED_SCRIPT_LEFT_COLS_ADVANCE_MS if self._timer_returns_game_time else constants.LEFT_COLS_ADVANCE_MS
                sync_target = max(0, action.time_ms - advance)
            else:
                advance = 0
                sync_target = action.time_ms

            if self._stop_event.is_set():
                return
            x, y = self._abs_pixel(grid[0], grid[1], side=not is_left_three_cols)

            if is_special:
                # 特殊单位到既定时间才出现，无法提前选中，先帧同步再选中执行
                await self._sync_to_frame(sync_target)
                if self._stop_event.is_set():
                    return
                self.action.select_operator_matchstick(x, y)
                await asyncio.sleep(1.0)
            else:
                # 普通单位保持原有流程：先选中进入子弹时间，再帧同步
                if not is_left_three_cols:
                    pos0 = self.pool.get_bar_index_pos(0)
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
                if self._stop_event.is_set():
                    return
                self.action.select_operator_matchstick(x, y)
                await asyncio.sleep(1.0)
                if self._stop_event.is_set():
                    return
                await self._sync_to_frame_after_select(action.time_ms)

            if self._stop_event.is_set():
                return
            if advance:
                self.timer.adjust(advance)
                self._notify_timer_adjusted()
                print(f"{constants.TIMER_ADJUST_MARKER}:{advance}")
            self.action.press_key(self.action.retreat_key())
            if action.operator_name and not action.is_object:
                self.pool.retreat(action.operator_name)
                # 若该干员绑定了召唤物，同步从部署区/已部署列表中清理
                bound_summon = self._summon_bindings_map.get(action.operator_name)
                if bound_summon:
                    self.pool.retreat(bound_summon)
                    self.pool.deactivate_summon(bound_summon)
                    if self.debug:
                        print(f"[执行] 干员 {action.operator_name} 撤退，清理绑定召唤物 {bound_summon}")

        elif action.action == ActionType.SKILL:
            # 解析目标格子，并识别特殊目标（干员名存在但无部署记录）
            grid = action.grid
            recorded_grid = None
            if action.operator_name and not action.is_object:
                recorded_grid = self.pool.get_deployed_grid(action.operator_name)
            if not grid:
                grid = recorded_grid
            if not grid:
                raise RuntimeError(f"技能操作缺少目标格子（干员/道具: {action.operator_name}）")
            is_special = action.operator_name is not None and recorded_grid is None and not action.is_object

            is_left_three_cols = grid[1] in (0, 1, 2)
            if is_left_three_cols:
                advance = constants.LOADED_SCRIPT_LEFT_COLS_ADVANCE_MS if self._timer_returns_game_time else constants.LEFT_COLS_ADVANCE_MS
                sync_target = max(0, action.time_ms - advance)
            else:
                advance = 0
                sync_target = action.time_ms

            if self._stop_event.is_set():
                return
            x, y = self._abs_pixel(grid[0], grid[1], side=not is_left_three_cols)

            if is_special:
                # 特殊单位到既定时间才出现，无法提前选中，先帧同步再选中执行
                await self._sync_to_frame(sync_target)
                if self._stop_event.is_set():
                    return
                self.action.select_operator_matchstick(x, y)
                await asyncio.sleep(1.0)
            else:
                # 普通单位保持原有流程：先选中进入子弹时间，再帧同步
                if not is_left_three_cols:
                    pos0 = self.pool.get_bar_index_pos(0)
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
                if self._stop_event.is_set():
                    return
                self.action.select_operator_matchstick(x, y)
                await asyncio.sleep(1.0)
                if self._stop_event.is_set():
                    return
                await self._sync_to_frame_after_select(action.time_ms)

            if self._stop_event.is_set():
                return
            if advance:
                self.timer.adjust(advance)
                self._notify_timer_adjusted()
                print(f"{constants.TIMER_ADJUST_MARKER}:{advance}")
            self.action.press_key(self.action.skill_key())

        elif action.action == ActionType.ADD_ITEM:
            if not action.operator_name or not action.grid:
                return
            bar_index = action.grid[0]
            charges = action.grid[1]
            self.pool.add_extra_item(action.operator_name, bar_index, charges)
            if self.script:
                # 同步回脚本数据，使后续读取的脚本状态保持一致
                existing = next((it for it in self.script.items if it.name == action.operator_name), None)
                if existing:
                    existing.charges = charges
                else:
                    from models.script_schema import ItemInfo
                    self.script.items.append(ItemInfo(name=action.operator_name, charges=charges))

        elif action.action == ActionType.ADD_SUMMON:
            if not action.operator_name:
                return
            summon = next((s for s in self.script.summons if s.name == action.operator_name), None)
            if summon is None:
                raise RuntimeError(f"脚本中未定义召唤物: {action.operator_name}")
            charges = 1
            if action.grid and len(action.grid) > 0:
                charges = max(1, int(action.grid[0]))
            self.pool.activate_summon(action.operator_name, charges)
            if self.debug:
                print(f"[执行] 召唤物 {action.operator_name} (费用 {summon.cost}) 加入部署栏 x{charges}")

        elif action.action == ActionType.REMOVE_SUMMON:
            if not action.operator_name:
                return
            summon = next((s for s in self.script.summons if s.name == action.operator_name), None)
            if summon is None:
                raise RuntimeError(f"脚本中未定义召唤物: {action.operator_name}")
            self.pool.deactivate_summon(action.operator_name)
            if self.debug:
                print(f"[执行] 召唤物 {action.operator_name} 从部署栏移除")

        elif action.action == ActionType.RESET_SUMMON:
            if not action.operator_name:
                return
            summon = next((s for s in self.script.summons if s.name == action.operator_name), None)
            if summon is None:
                raise RuntimeError(f"脚本中未定义召唤物: {action.operator_name}")
            target_count = 1
            if action.grid and len(action.grid) > 0:
                target_count = max(0, int(action.grid[0]))
            # 强制修正部署栏数量：已部署召唤物视为强制返回并合并为指定数量
            if action.operator_name in self.pool._deployed:
                self.pool.retreat(action.operator_name)
            self.pool.set_summon_charges(action.operator_name, target_count)
            if self.debug:
                print(f"[执行] 召唤物 {action.operator_name} 强制修正数量为 {target_count}")

        elif action.action == ActionType.SPEED_UP:
            self.action.press_key(self.action.speed_key())
        elif action.action == ActionType.SPEED_DOWN:
            self.action.press_key("1")
        elif action.action == ActionType.PAUSE:
            self.action.press_key(self.action.pause_key())

    def _fmt_time(self, time_ms: Optional[int] = None) -> str:
        t = self.timer.get_elapsed_ms() if time_ms is None else time_ms
        if self.cost_sync is None:
            return f"timer={t:.1f}ms"
        count = self.cost_sync.white_pixel_count()
        if count is None:
            return f"timer={t:.1f}ms"
        game = self._estimate_game_time_at_count(count, t)
        if game is None:
            return f"timer={t:.1f}ms"
        return f"timer={t:.1f}ms game≈{game:.1f}ms"

    def _frames_behind(self, current_frame: int, target_frame: int, time_ms: int) -> int:
        """计算当前帧比目标帧落后多少帧（0 表示重合或超前）。"""
        if self.cost_sync is None:
            return 0
        if hasattr(self.cost_sync, "get_calibration"):
            cycle = self.cost_sync.get_calibration(self._game_time_ms(time_ms)).cycle_length
        else:
            cycle = self.cost_sync.cycle_length
        return (target_frame - current_frame) % cycle

    async def _execute_action(self, action: OperatorAction):
        """单 action 执行，包含完整的暂停/恢复外壳。"""
        import pydirectinput
        if self.debug:
            print(f"[执行] time={self.timer.get_elapsed_ms()}ms, 目标={action.time_ms}ms, action={action.action.value} {action.operator_name}")
        pause_key = self.action.pause_key()

        if self.debug:
            print(f"[动作计时] 开始暂停外壳: 目标={action.time_ms}ms, timer={self.timer.get_elapsed_ms():.1f}ms")

        # 临时诊断：测量按下暂停键期间游戏内时间前进了多少
        if self.debug and self.cost_sync is not None:
            _t0 = self.timer.get_elapsed_ms()
            _count0 = self.cost_sync.white_pixel_count()
            _game0 = self._estimate_game_time_at_count(_count0, _t0) if _count0 is not None else None
        else:
            _game0 = None

        # 关键：到达目标后立刻按下暂停，避免前面的截图/计算耗时让游戏继续跑
        pydirectinput.keyDown(pause_key)
        self.timer.pause()
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)

        # 暂停后再做详细帧校验和 fmt_time，这些操作不再影响实际动作时机
        if self.debug:
            if self.cost_sync is not None:
                count = self.cost_sync.white_pixel_count()
                game_time_ms = self._game_time_ms(action.time_ms)
                target_frame = self.cost_sync.target_frame_index(game_time_ms)
                current_frame = self.cost_sync.current_frame(count, game_time_ms)
                print(f"[执行帧校验] 目标帧={target_frame}, 当前帧={current_frame}, 白像素={count}")
            print(f"[动作计时] 暂停后: 目标={action.time_ms}ms, {self._fmt_time()}")

        if self.debug and self.cost_sync is not None and _game0 is not None:
            _t1 = self.timer.get_elapsed_ms()
            _count1 = self.cost_sync.white_pixel_count()
            _game1 = self._estimate_game_time_at_count(_count1, _t1) if _count1 is not None else None
            if _game1 is not None:
                _speed = 2 if self._speed2x_ref_ms is not None else 1
                print(
                    f"[按键延迟测量] 按暂停前 timer={_t0}ms game≈{_game0:.1f}ms, "
                    f"按暂停后 timer={_t1}ms game≈{_game1:.1f}ms, "
                    f"游戏内前进={_game1 - _game0:.1f}ms, 预期前进={_speed * (_t1 - _t0):.1f}ms"
                )

        # 二倍速下根据费用条修正计时器，避免暂停延迟导致误差累积
        self._resync_timer_at_pause(action.time_ms)
        if self.debug:
            print(f"[动作计时] resync后: {self._fmt_time()}")

        await asyncio.sleep(1.0)

        try:
            if self.debug:
                print(f"[动作计时] 执行核心前: 目标={action.time_ms}ms, {self._fmt_time()}")
            await self._execute_action_core(action)
            if self.debug:
                print(f"[动作计时] 执行核心后: 目标={action.time_ms}ms, {self._fmt_time()}")
            if action.action == ActionType.PAUSE:
                # PAUSE 把游戏从暂停切回运行，计时器同步恢复
                self.timer.resume()
        finally:
            if action.action != ActionType.PAUSE:
                await asyncio.sleep(1.0)
                pydirectinput.keyDown(pause_key)
                self.timer.resume()
                await asyncio.sleep(0.05)
                pydirectinput.keyUp(pause_key)
                await asyncio.sleep(0.05)

    async def _execute_batch(self, batch: List[OperatorAction]):
        """批量执行同 time_ms 的操作，只暂停/恢复一次游戏。"""

        import pydirectinput
        pause_key = self.action.pause_key()

        # 临时诊断：测量按下暂停键期间游戏内时间前进了多少
        _game0 = None
        if self.debug and self.cost_sync is not None and batch:
            _t0 = self.timer.get_elapsed_ms()
            _count0 = self.cost_sync.white_pixel_count()
            _game0 = self._estimate_game_time_at_count(_count0, _t0) if _count0 is not None else None

        pydirectinput.keyDown(pause_key)
        self.timer.pause()
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)

        if self.debug and self.cost_sync is not None and _game0 is not None:
            _t1 = self.timer.get_elapsed_ms()
            _count1 = self.cost_sync.white_pixel_count()
            _game1 = self._estimate_game_time_at_count(_count1, _t1) if _count1 is not None else None
            if _game1 is not None:
                _speed = 2 if self._speed2x_ref_ms is not None else 1
                print(
                    f"[按键延迟测量-批量] 按暂停前 timer={_t0}ms game≈{_game0:.1f}ms, "
                    f"按暂停后 timer={_t1}ms game≈{_game1:.1f}ms, "
                    f"游戏内前进={_game1 - _game0:.1f}ms, 预期前进={_speed * (_t1 - _t0):.1f}ms"
                )

        # 二倍速下根据费用条修正计时器，避免暂停延迟导致误差累积
        if batch:
            self._resync_timer_at_pause(batch[0].time_ms)
        if self.debug:
            print(f"[动作计时-批量] resync后: {self._fmt_time()}")

        await asyncio.sleep(1.0)

        try:
            if self.debug:
                print(f"[动作计时-批量] 执行核心前: 目标={batch[0].time_ms}ms, {self._fmt_time()}")
                action = batch[0]
                count = self.cost_sync.white_pixel_count()
                game_time_ms = self._game_time_ms(action.time_ms)
                target_frame = self.cost_sync.target_frame_index(game_time_ms)
                current_frame = self.cost_sync.current_frame(count, game_time_ms)
                print(f"[批量执行帧校验] 目标帧={target_frame}, 当前帧={current_frame}, 白像素={count}")
            for idx, action in enumerate(batch):
                if self._stop_event.is_set():
                    break
                print(f"[批量执行] 第 {idx+1}/{len(batch)} 个: {action.action} {action.operator_name}")
                await self._execute_action_core(action)
                print(f"[批量执行] 第 {idx+1}/{len(batch)} 个完成")
                # 同 batch 内操作之间留 1.0s 让游戏 UI 稳定，避免连续拖拽冲突
                if idx < len(batch) - 1:
                    await asyncio.sleep(1.0)
            if self.debug:
                print(f"[动作计时-批量] 执行核心后: 目标={batch[0].time_ms}ms, {self._fmt_time()}")
        finally:
            await asyncio.sleep(1.0)
            pydirectinput.keyDown(pause_key)
            self.timer.resume()
            await asyncio.sleep(0.05)
            pydirectinput.keyUp(pause_key)
            await asyncio.sleep(0.05)

    def _build_execution_units(self) -> List:
        """把脚本按时间聚类：同 time_ms 的操作合成 batch，差距 <40ms 的 batch 合成 cluster。"""
        if not self.script:
            return []
        actions = self.script.actions
        # 先按 actual_target 拆成 batch（PAUSE 单独成 batch）
        batches: List[Tuple[int, List[OperatorAction]]] = []
        i = 0
        while i < len(actions):
            action = actions[i]
            actual_target = self._get_actual_target(action)
            batch = [action]
            if action.action != ActionType.PAUSE:
                while (
                    i + 1 < len(actions)
                    and self._get_actual_target(actions[i + 1]) == actual_target
                    and actions[i + 1].action != ActionType.PAUSE
                ):
                    i += 1
                    batch.append(actions[i])
            batches.append((actual_target, batch))
            i += 1

        # 再把相邻 batch 聚类：actual_target 差距 < 40ms 且都不是 PAUSE
        units: List = []
        cluster: List[List[OperatorAction]] = []
        prev_target: Optional[int] = None
        for target, batch in batches:
            if not cluster:
                cluster.append(batch)
                prev_target = target
                continue
            prev_is_pause = cluster[-1][0].action == ActionType.PAUSE
            cur_is_pause = batch[0].action == ActionType.PAUSE
            if not prev_is_pause and not cur_is_pause and target - prev_target < 40:
                cluster.append(batch)
                prev_target = target
            else:
                if len(cluster) == 1:
                    units.append(("batch", cluster[0]))
                else:
                    units.append(("cluster", cluster))
                cluster = [batch]
                prev_target = target
        if cluster:
            if len(cluster) == 1:
                units.append(("batch", cluster[0]))
            else:
                units.append(("cluster", cluster))
        return units

    async def _execute_cluster(self, groups: List[List[OperatorAction]]):
        """在单个暂停外壳中依次执行多组时间紧贴的操作，组间用 p_and_esc_click 推进一帧。"""
        import pydirectinput

        pause_key = self.action.pause_key()
        pydirectinput.keyDown(pause_key)
        self.timer.pause()
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)

        # 二倍速下根据费用条修正计时器，避免暂停延迟导致误差累积
        if groups and groups[0]:
            self._resync_timer_at_pause(groups[0][0].time_ms)
        if self.debug:
            print(f"[动作计时-聚类] resync后: {self._fmt_time()}")

        await asyncio.sleep(1.0)

        try:
            if self.debug:
                print(f"[动作计时-聚类] 执行核心前: 目标={groups[0][0].time_ms}ms, {self._fmt_time()}")
                action = groups[0][0]
                count = self.cost_sync.white_pixel_count()
                game_time_ms = self._game_time_ms(action.time_ms)
                target_frame = self.cost_sync.target_frame_index(game_time_ms)
                current_frame = self.cost_sync.current_frame(count, game_time_ms)
                print(f"[聚类执行帧校验] 目标帧={target_frame}, 当前帧={current_frame}, 白像素={count}")
            for gi, group in enumerate(groups):
                if self._stop_event.is_set():
                    break
                print(f"[聚类执行] 第 {gi + 1}/{len(groups)} 组, 共 {len(group)} 个操作")
                for idx, action in enumerate(group):
                    await self._execute_action_core(action)
                    if idx < len(group) - 1:
                        await asyncio.sleep(1.0)
                # 不是最后一组时推进一帧（33ms），保持暂停状态
                if gi < len(groups) - 1:
                    pos0 = self.pool.get_bar_index_pos(0)
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
                    delta = self._skip_one_frame_and_measure(groups[0][0].time_ms)
                    self.timer.adjust(delta)
                    self._notify_timer_adjusted()
                    print(f"{constants.TIMER_ADJUST_MARKER}:{delta}")
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
            if self.debug:
                print(f"[动作计时-聚类] 执行核心后: 目标={groups[0][0].time_ms}ms, {self._fmt_time()}")
        finally:
            await asyncio.sleep(1.0)
            pydirectinput.keyDown(pause_key)
            self.timer.resume()
            await asyncio.sleep(0.05)
            pydirectinput.keyUp(pause_key)
            await asyncio.sleep(0.05)

    async def run(self):
        if self.script is None:
            raise RuntimeError("未加载脚本")
        self._stop_event.clear()

        units = self._build_execution_units()
        try:
            for kind, payload in units:
                if self._stop_event.is_set():
                    break
                if kind == "batch":
                    batch = payload
                    actual_target = self._get_actual_target(batch[0])
                    actual_target = self._wait_target_ms(actual_target, batch[0])
                    if self.debug:
                        print(f"[动作计时] 等待目标: 原始={batch[0].time_ms}ms 实际={actual_target}ms, {self._fmt_time()}")
                    ok = await self.wait_until(actual_target)
                    if not ok:
                        break
                    if len(batch) == 1:
                        await self._execute_action(batch[0])
                    else:
                        print(f"[批量执行] time={actual_target}ms, 共 {len(batch)} 个操作")
                        await self._execute_batch(batch)
                else:  # cluster
                    groups = payload
                    actual_target = self._get_actual_target(groups[0][0])
                    actual_target = self._wait_target_ms(actual_target, groups[0][0])
                    if self.debug:
                        print(f"[动作计时] 等待目标(聚类): 原始={groups[0][0].time_ms}ms 实际={actual_target}ms, {self._fmt_time()}")
                    ok = await self.wait_until(actual_target)
                    if not ok:
                        break
                    print(f"[聚类执行] time={actual_target}ms, 共 {len(groups)} 组")
                    await self._execute_cluster(groups)
        finally:
            self.timer.reset()

    def stop(self):
        self._stop_event.set()
        self.timer.reset()

    def pause(self):
        self.timer.pause()

    def resume(self):
        self.timer.resume()

    def get_state(self) -> ExecutorState:
        return ExecutorState(
            is_running=self.timer._running and not self.timer._paused,
            is_paused=self.timer._paused,
            current_time_ms=self.timer.get_elapsed_ms(),
            stage_code=self.script.stage_code if self.script else None,
        )
