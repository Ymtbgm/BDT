import asyncio
import os
import time
from typing import Optional, Dict, List, Tuple, Union

import cv2
import numpy as np
from pydantic import BaseModel

from core.capture import WindowCapture
from core.grid_mapper import GridMapper
from core.timer import StageTimer
from core.ocr_engine import OCREngine
from core.operator_pool import OperatorPool
from core.cost_bar_sync import CostBarSync
from core.cost_bar_sync_cc import CostBarSyncCC
import core.constants as constants
from models.script_schema import ScriptModel, ActionType, OperatorAction


CostBarSyncType = Union[CostBarSync, CostBarSyncCC]


class ExecutorState(BaseModel):
    is_running: bool = False
    is_paused: bool = False
    current_time_ms: int = 0
    stage_name: Optional[str] = None


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

    def set_cost_sync(self, cost_sync: Optional[CostBarSyncType]):
        self.cost_sync = cost_sync
        self._cost_bar_maxed = False
    def set_speed2x_reference(self, ref_ms: Optional[int] = None):
        """设置二倍速时间压缩的参考点，用于费用条同步时反推原始游戏内时间。"""
        self._speed2x_ref_ms = ref_ms

    def _game_time_ms(self, time_ms: int) -> int:
        """将压缩后的现实时间还原为原始游戏内时间，供费用条帧同步使用。"""
        if self._speed2x_ref_ms is not None and time_ms > self._speed2x_ref_ms:
            return self._speed2x_ref_ms + (time_ms - self._speed2x_ref_ms) * 2
        return time_ms

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
        if self._speed2x_ref_ms is None or self.cost_sync is None:
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
        ref = self._speed2x_ref_ms
        new_timer_ms = int(ref + (actual_game_ms - ref) / 2)
        old_timer_ms = self.timer.get_elapsed_ms()
        self.timer.adjust(new_timer_ms - old_timer_ms)
        if self.debug:
            print(
                f"[二倍速计时器重同步] 压缩目标={compressed_time}ms, "
                f"反推游戏时间={actual_game_ms:.1f}ms, "
                f"旧计时器={old_timer_ms}ms, 新计时器={new_timer_ms}ms"
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
        script.sort_actions()
        w, h = self.capture.get_window_size()
        self.grid = GridMapper(
            w, h, script.grid_rows, script.grid_cols,
            stage_code=script.stage_code, stage_name=script.stage_name,
        )
        support_count = 1 if borrow_support else 0
        self.pool = OperatorPool(
            w, h, script.operators, script.items, script.summons,
            support_count=support_count,
        )
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        self.pool.set_window_offset(left, top)

    def verify_stage_name(self) -> bool:
        if not self.script or not self.script.stage_name:
            return True
        frame = self.capture.capture()
        found = self.ocr.find_text(frame, self.script.stage_name)
        if found is None:
            return False
        return True

    async def wait_until(self, target_ms: int, check_interval: float = 0.005):
        while self.timer.get_elapsed_ms() < target_ms:
            if self._stop_event.is_set():
                return False
            # 剩余时间 > 5ms 时用 asyncio.sleep 避免空转；
            # 最后 5ms 自旋等待，消除 sleep 精度抖动（Windows 默认 ~15ms）
            if target_ms - self.timer.get_elapsed_ms() > constants.WAIT_SPIN_THRESHOLD_MS:
                await asyncio.sleep(check_interval)
        return True

    def _get_actual_target(self, action: OperatorAction) -> int:
        """对最左三列的 RETREAT/SKILL 提前触发。"""
        if action.action not in (ActionType.RETREAT, ActionType.SKILL):
            return action.time_ms
        grid = action.grid
        if not grid and action.operator_name and not action.is_object:
            grid = self.pool.get_deployed_grid(action.operator_name)
        if grid and grid[1] in (0, 1, 2):
            return max(0, action.time_ms - constants.LEFT_COLS_ADVANCE_MS)
        return action.time_ms

    def _wait_target_ms(self, target_ms: int, action: OperatorAction) -> int:
        """二倍速下将 wait_until 目标提前，抵消暂停键延迟导致的触发偏晚。"""
        if (
            self._speed2x_ref_ms is not None
            and action.action != ActionType.PAUSE
            and target_ms > self._speed2x_ref_ms
        ):
            return max(self._speed2x_ref_ms, target_ms - constants.TWOX_EARLY_TRIGGER_MS)
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
        costs = self._recognize_operator_costs()
        if costs:
            self.pool.set_operator_costs(costs)
            if self.debug:
                print(f"[部署栏OCR] 首个动作暂停时已设置费用: {costs}")
        elif self.debug:
            print("[部署栏OCR] 首个动作暂停时未识别到费用，使用初始序号排序")

    def _recognize_operator_costs(self) -> Dict[str, int]:
        """按干员格子精确裁剪并 OCR 识别费用，返回 {operator_name: cost}。

        识别失败或数量不匹配时返回空字典，调用方应回退到按初始序号排序。
        """
        if not self.script or not self.script.operators:
            return {}
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        ratios = constants.DEPLOY_BAR_COST_ROI_RATIOS
        y = int(h * ratios[1]) + top
        rh = int(h * ratios[3])

        support_count = 1 if getattr(self, "borrow_support", False) else 0
        operators_to_recognize = self.script.operators[:-support_count] if support_count > 0 else self.script.operators
        num_operators = len(self.script.operators)
        num_items = len(self.script.items) if self.script.items else 0
        total = num_operators + num_items
        cell_w = w / 12 if total <= 12 else w / total

        session_id = int(time.time() * 1000)
        session_dir = os.path.join("debug", "operator_cost_ocr", str(session_id))
        if self.debug:
            os.makedirs(session_dir, exist_ok=True)
            print(
                f"[部署栏OCR] 会话={session_id} 窗口=({w}x{h}) 格子宽={cell_w:.1f} "
                f"干员数={num_operators} 道具数={num_items} 助战={support_count}"
            )

        mapping = {}
        for i, name in enumerate(operators_to_recognize):
            bar_index = total - 1 - i
            cx = w - cell_w * (bar_index + 0.5)
            x = int(cx) + left
            rw = 53
            try:
                img = self.capture.capture_roi(x, y, rw, rh)
            except Exception as e:
                if self.debug:
                    print(f"[部署栏OCR] {name}: 截取 ROI 失败: {e}")
                continue

            raw_path = os.path.join(session_dir, f"{name}_raw.png") if self.debug else None
            fixed_path = os.path.join(session_dir, f"{name}_fixed.png") if self.debug else None
            inv_path = os.path.join(session_dir, f"{name}_inv.png") if self.debug else None

            fixed_img = self._preprocess_cost_image(img)
            inv_img = self._preprocess_cost_image_inv(img)

            # 固定阈值二值化 → 反色二值化
            fixed_result = self._extract_cost_with_conf(
                self.ocr.recognize(fixed_img, min_confidence=0.5), min_conf=0.5
            )
            inv_result = self._extract_cost_with_conf(
                self.ocr.recognize(inv_img, min_confidence=0.5), min_conf=0.5
            )

            if self.debug:
                os.makedirs(session_dir, exist_ok=True)
                cv2.imwrite(raw_path, img)
                cv2.imwrite(fixed_path, fixed_img)
                cv2.imwrite(inv_path, inv_img)
                fixed_str = f"{fixed_result[0]}({fixed_result[1]:.2f})" if fixed_result else "失败"
                inv_str = f"{inv_result[0]}({inv_result[1]:.2f})" if inv_result else "失败"

            chosen = None
            chosen_source = None
            if fixed_result:
                chosen = fixed_result[0]
                chosen_source = "固定阈值"
            elif inv_result:
                chosen = inv_result[0]
                chosen_source = "反色"

            if chosen is not None:
                mapping[name] = chosen
                if self.debug:
                    print(f"[部署栏OCR] {name}: 固定阈值={fixed_str}, 反色={inv_str} → {chosen} ({chosen_source})")
            elif self.debug:
                print(f"[部署栏OCR] {name}: 固定阈值={fixed_str}, 反色={inv_str} → 失败")

        expected = len(operators_to_recognize)
        if len(mapping) < expected:
            if self.debug:
                print(
                    f"[部署栏OCR] 仅识别到 {len(mapping)}/{expected} 个费用，"
                    f"回退到初始序号排序"
                )
            return {}

        # 自动检测手动借用的助战干员：仅直接开始作战时启用。
        # 直接开始作战未勾选 borrow_support，但用户可能手动借用。
        # 若最右侧干员费用低于其左侧第一名干员，则判定最右侧为助战，固定在最右不参与排序。
        if (
            self.direct_start
            and support_count == 0
            and len(self.script.operators) >= 2
        ):
            rightmost_name = self.script.operators[-1]
            left_neighbor_name = self.script.operators[-2]
            if rightmost_name in mapping and left_neighbor_name in mapping:
                if mapping[rightmost_name] < mapping[left_neighbor_name]:
                    self.borrow_support = True
                    if self.pool is not None:
                        self.pool.set_support_count(1)
                    if self.debug:
                        print(
                            f"[部署栏OCR] 检测到手动助战: {rightmost_name}"
                            f"({mapping[rightmost_name]}) < {left_neighbor_name}"
                            f"({mapping[left_neighbor_name]}), 已固定为最右"
                        )

        if self.debug:
            print(f"[部署栏OCR] 识别费用: {mapping}")
        return mapping

    def _extract_cost_with_conf(self, results: list, min_conf: float = 0.5) -> Optional[Tuple[int, float]]:
        """从 OCR 结果中提取置信度最高的纯数字费用，无有效结果返回 None。"""
        best_cost = None
        best_conf = 0.0
        for bbox, (text, conf) in results:
            if conf < min_conf:
                continue
            digits = "".join(c for c in text if c.isdigit())
            if not digits:
                continue
            try:
                cost = int(digits)
            except ValueError:
                continue
            if conf > best_conf:
                best_conf = conf
                best_cost = cost
        return (best_cost, best_conf) if best_cost is not None else None

    def _extract_cost_from_results(self, results: list) -> Optional[int]:
        """从单格 OCR 结果中提取最可信的纯数字费用，无有效结果返回 None。"""
        res = self._extract_cost_with_conf(results, min_conf=constants.DEPLOY_BAR_COST_CONFIDENCE)
        return res[0] if res else None

    def _preprocess_cost_image(self, img: np.ndarray) -> np.ndarray:
        """预处理费用数字截图：放大后固定阈值二值化并轻微闭运算，强化白字。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(
            gray,
            constants.DEPLOY_BAR_COST_WHITE_THRESHOLD,
            255,
            cv2.THRESH_BINARY,
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    def _preprocess_cost_image_inv(self, img: np.ndarray) -> np.ndarray:
        """反色二值化（黑字白底），作为固定阈值失败时的回退。"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    async def _advance_frame_in_bullet_time(self):
        """进入子弹时间后调用 p_and_esc_click 推进一帧，再退出子弹时间。

        注意：此函数用于把游戏画面修正到计时器已暂停的理论时间，
        因此不调整计时器；计时器仍保持在暂停时的目标时间点。
        """
        pos0 = self.pool.get_bar_index_pos(0)
        if pos0:
            self.action.select_at(pos0[0], pos0[1])
            await asyncio.sleep(1.0)
        self.action.p_and_esc_click()
        if pos0:
            self.action.select_at(pos0[0], pos0[1])
            await asyncio.sleep(1.0)

    async def _sync_to_frame(self, time_ms: int):
        """基于费用条白像素进行帧同步，最多跳 1 帧。

        流程：
          1. 若当前帧与目标帧差距 > 2，说明费用条可能已满/失准，直接执行；
          2. 若当前白像素匹配目标帧或下一帧，直接执行；
          3. 否则进入子弹时间跳 1 帧，再次匹配目标帧；匹配不到也直接执行。
        """
        if self.cost_sync is None:
            return

        game_time_ms = self._game_time_ms(time_ms)
        target_frame = self.cost_sync.target_frame_index(game_time_ms)
        count = self.cost_sync.white_pixel_count()
        if count is None:
            return

        current_frame = self.cost_sync.current_frame(count, game_time_ms)
        distance = self.cost_sync.frame_distance(current_frame, target_frame)
        if self.cost_sync.debug:
            print(
                f"[费用条同步] 目标帧={target_frame}, 当前帧={current_frame}, "
                f"白像素={count}, 帧差={distance}"
            )

        # 差距过大时直接执行，避免费用条满后的误伤
        if distance >= 2:
            if self.cost_sync.debug:
                print("[费用条同步] 帧差 >= 2，跳过同步直接执行")
            return

        # 已匹配目标帧或下一帧，直接执行
        cycle = self.cost_sync.get_calibration(game_time_ms).cycle_length
        if self.cost_sync.is_match(count, target_frame, game_time_ms) or self.cost_sync.is_match(
            count, (target_frame + 1) % cycle, game_time_ms
        ):
            if self.cost_sync.debug:
                print("[费用条同步] 已匹配目标帧/下一帧，直接执行")
            return

        # 跳 1 帧后再匹配目标帧，匹配不到也直接执行
        await self._advance_frame_in_bullet_time()
        count = self.cost_sync.white_pixel_count()
        if self.cost_sync.debug:
            current_frame = self.cost_sync.current_frame(count, game_time_ms)
            print(f"[费用条同步] 跳 1 帧后: 当前帧={current_frame}, 白像素={count}")

    async def _sync_to_frame_after_select(self, time_ms: int):
        """在已选中干员（子弹时间）后进行帧同步，直接 p_and_esc_click 跳帧，不再进出子弹时间。"""
        if self.cost_sync is None:
            return

        game_time_ms = self._game_time_ms(time_ms)
        target_frame = self.cost_sync.target_frame_index(game_time_ms)
        count = self.cost_sync.white_pixel_count()
        if count is None:
            return

        current_frame = self.cost_sync.current_frame(count, game_time_ms)
        distance = self.cost_sync.frame_distance(current_frame, target_frame)
        if self.cost_sync.debug:
            print(
                f"[费用条同步-选中后] 目标帧={target_frame}, 当前帧={current_frame}, "
                f"白像素={count}, 帧差={distance}"
            )

        if distance >= 2:
            if self.cost_sync.debug:
                print("[费用条同步-选中后] 帧差 >= 2，跳过同步直接执行")
            return

        cycle = self.cost_sync.get_calibration(game_time_ms).cycle_length
        if self.cost_sync.is_match(count, target_frame, game_time_ms) or self.cost_sync.is_match(
            count, (target_frame + 1) % cycle, game_time_ms
        ):
            if self.cost_sync.debug:
                print("[费用条同步-选中后] 已匹配目标帧/下一帧，直接执行")
            return

        self.action.p_and_esc_click()
        count = self.cost_sync.white_pixel_count()
        if self.cost_sync.debug:
            current_frame = self.cost_sync.current_frame(count, game_time_ms)
            print(f"[费用条同步-选中后] 跳 1 帧后: 当前帧={current_frame}, 白像素={count}")

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
            grid = action.grid
            if action.operator_name and not grid and not action.is_object:
                grid = self.pool.get_deployed_grid(action.operator_name)
            if not grid:
                raise RuntimeError(f"撤退操作缺少目标格子（干员/道具: {action.operator_name}）")

            is_left_three_cols = grid[1] in (0, 1, 2)
            if not is_left_three_cols:
                pos0 = self.pool.get_bar_index_pos(0)
                if pos0:
                    self.action.select_at(pos0[0], pos0[1])
                    await asyncio.sleep(1.0)

            if self._stop_event.is_set():
                return
            x, y = self._abs_pixel(grid[0], grid[1], side=not is_left_three_cols)
            self.action.select_operator_matchstick(x, y)
            await asyncio.sleep(1.0)
            if self._stop_event.is_set():
                return
            await self._sync_to_frame_after_select(action.time_ms)
            if is_left_three_cols:
                self.timer.adjust(18.0)
                print("__TIMER_ADJUST__:18.0")
            self.action.press_key(self.action.retreat_key())
            if action.operator_name and not action.is_object:
                self.pool.retreat(action.operator_name)

        elif action.action == ActionType.SKILL:
            grid = action.grid
            if action.operator_name and not grid and not action.is_object:
                grid = self.pool.get_deployed_grid(action.operator_name)
            if not grid:
                raise RuntimeError(f"技能操作缺少目标格子（干员/道具: {action.operator_name}）")

            is_left_three_cols = grid[1] in (0, 1, 2)
            if not is_left_three_cols:
                pos0 = self.pool.get_bar_index_pos(0)
                if pos0:
                    self.action.select_at(pos0[0], pos0[1])
                    await asyncio.sleep(1.0)

            if self._stop_event.is_set():
                return
            x, y = self._abs_pixel(grid[0], grid[1], side=not is_left_three_cols)
            self.action.select_operator_matchstick(x, y)
            await asyncio.sleep(1.0)
            if self._stop_event.is_set():
                return
            await self._sync_to_frame_after_select(action.time_ms)
            if is_left_three_cols:
                self.timer.adjust(18.0)
                print("__TIMER_ADJUST__:18.0")
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

        elif action.action == ActionType.SPEED_UP:
            self.action.press_key(self.action.speed_key())
        elif action.action == ActionType.SPEED_DOWN:
            self.action.press_key("1")
        elif action.action == ActionType.PAUSE:
            self.action.press_key(self.action.pause_key())

    async def _execute_action(self, action: OperatorAction):
        """单 action 执行，包含完整的暂停/恢复外壳。"""
        import pydirectinput
        if self.debug:
            print(f"[执行] time={self.timer.get_elapsed_ms()}ms, 目标={action.time_ms}ms, action={action.action.value} {action.operator_name}")
            if self.cost_sync is not None:
                count = self.cost_sync.white_pixel_count()
                game_time_ms = self._game_time_ms(action.time_ms)
                target_frame = self.cost_sync.target_frame_index(game_time_ms)
                current_frame = self.cost_sync.current_frame(count, game_time_ms)
                print(f"[执行帧校验] 目标帧={target_frame}, 当前帧={current_frame}, 白像素={count}")
        pause_key = self.action.pause_key()

        # 临时诊断：测量按下暂停键期间游戏内时间前进了多少
        if self.debug and self.cost_sync is not None:
            _t0 = self.timer.get_elapsed_ms()
            _count0 = self.cost_sync.white_pixel_count()
            _game0 = self._estimate_game_time_at_count(_count0, _t0) if _count0 is not None else None
        else:
            _game0 = None

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
                    f"[按键延迟测量] 按暂停前 timer={_t0}ms game≈{_game0:.1f}ms, "
                    f"按暂停后 timer={_t1}ms game≈{_game1:.1f}ms, "
                    f"游戏内前进={_game1 - _game0:.1f}ms, 预期前进={_speed * (_t1 - _t0):.1f}ms"
                )

        # 二倍速下根据费用条修正计时器，避免暂停延迟导致误差累积
        self._resync_timer_at_pause(action.time_ms)

        await asyncio.sleep(1.0)

        try:
            await self._execute_action_core(action)
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

        await asyncio.sleep(1.0)

        try:
            if self.debug and self.cost_sync is not None and batch:
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

        await asyncio.sleep(1.0)

        try:
            if self.debug and self.cost_sync is not None and groups and groups[0]:
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
                    self.action.p_and_esc_click()
                    self.timer.adjust(constants.ADVANCE_FRAME_MS)
                    print(f"{constants.TIMER_ADJUST_MARKER}:{constants.ADVANCE_FRAME_MS}")
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
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
            stage_name=self.script.stage_name if self.script else None,
        )
