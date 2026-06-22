import asyncio
import time
from typing import Optional, Dict, List, Tuple
from pydantic import BaseModel

from core.capture import WindowCapture
from core.grid_mapper import GridMapper
from core.timer import StageTimer
from core.ocr_engine import OCREngine
from core.operator_pool import OperatorPool
from core.cost_bar_sync import CostBarSync
from models.script_schema import ScriptModel, ActionType, OperatorAction


class ExecutorState(BaseModel):
    is_running: bool = False
    is_paused: bool = False
    current_time_ms: int = 0
    stage_name: Optional[str] = None


class ScriptExecutor:
    def __init__(self, capture: WindowCapture, ocr: OCREngine, action_module):
        self.capture = capture
        self.ocr = ocr
        self.action = action_module
        self.timer = StageTimer()
        self.script: Optional[ScriptModel] = None
        self.grid: Optional[GridMapper] = None
        self.pool: Optional[OperatorPool] = None
        self.cost_sync: Optional[CostBarSync] = None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()

    def set_cost_sync(self, cost_sync: Optional[CostBarSync]):
        self.cost_sync = cost_sync

    def load_script(self, script: ScriptModel):
        self.script = script
        script.sort_actions()
        w, h = self.capture.get_window_size()
        self.grid = GridMapper(
            w, h, script.grid_rows, script.grid_cols,
            stage_code=script.stage_code, stage_name=script.stage_name,
        )
        self.pool = OperatorPool(w, h, script.operators, script.items)
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
            if target_ms - self.timer.get_elapsed_ms() > 5:
                await asyncio.sleep(check_interval)
        return True

    def _get_actual_target(self, action: OperatorAction) -> int:
        """对最左三列的 RETREAT/SKILL 提前 18ms 触发。"""
        if action.action not in (ActionType.RETREAT, ActionType.SKILL):
            return action.time_ms
        grid = action.grid
        if not grid and action.operator_name and not action.is_object:
            grid = self.pool.get_deployed_grid(action.operator_name)
        if grid and grid[1] in (0, 1, 2):
            return max(0, action.time_ms - 18)
        return action.time_ms

    def _abs_pixel(self, row: int, col: int, side: bool = False):
        x, y = self.grid.grid_to_pixel(row, col, side=side)
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return x + left, y + top

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

        target_frame = self.cost_sync.target_frame_index(time_ms)
        count = self.cost_sync.white_pixel_count()
        if count is None:
            return

        current_frame = self.cost_sync.current_frame(count)
        distance = CostBarSync.frame_distance(current_frame, target_frame)
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
        if self.cost_sync.is_match(count, target_frame) or self.cost_sync.is_match(
            count, (target_frame + 1) % 30
        ):
            if self.cost_sync.debug:
                print("[费用条同步] 已匹配目标帧/下一帧，直接执行")
            return

        # 跳 1 帧后再匹配目标帧，匹配不到也直接执行
        await self._advance_frame_in_bullet_time()
        count = self.cost_sync.white_pixel_count()
        if self.cost_sync.debug:
            current_frame = self.cost_sync.current_frame(count)
            print(f"[费用条同步] 跳 1 帧后: 当前帧={current_frame}, 白像素={count}")

    async def _sync_to_frame_after_select(self, time_ms: int):
        """在已选中干员（子弹时间）后进行帧同步，直接 p_and_esc_click 跳帧，不再进出子弹时间。"""
        if self.cost_sync is None:
            return

        target_frame = self.cost_sync.target_frame_index(time_ms)
        count = self.cost_sync.white_pixel_count()
        if count is None:
            return

        current_frame = self.cost_sync.current_frame(count)
        distance = CostBarSync.frame_distance(current_frame, target_frame)
        if self.cost_sync.debug:
            print(
                f"[费用条同步-选中后] 目标帧={target_frame}, 当前帧={current_frame}, "
                f"白像素={count}, 帧差={distance}"
            )

        if distance >= 2:
            if self.cost_sync.debug:
                print("[费用条同步-选中后] 帧差 >= 2，跳过同步直接执行")
            return

        if self.cost_sync.is_match(count, target_frame) or self.cost_sync.is_match(
            count, (target_frame + 1) % 30
        ):
            if self.cost_sync.debug:
                print("[费用条同步-选中后] 已匹配目标帧/下一帧，直接执行")
            return

        self.action.p_and_esc_click()
        count = self.cost_sync.white_pixel_count()
        if self.cost_sync.debug:
            current_frame = self.cost_sync.current_frame(count)
            print(f"[费用条同步-选中后] 跳 1 帧后: 当前帧={current_frame}, 白像素={count}")

    async def _execute_action_core(self, action: OperatorAction):
        """仅执行操作逻辑，不处理暂停/恢复外壳。"""
        if self._stop_event.is_set():
            return
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

        elif action.action == ActionType.SPEED_UP:
            self.action.press_key("2")
        elif action.action == ActionType.SPEED_DOWN:
            self.action.press_key("1")
        elif action.action == ActionType.PAUSE:
            self.action.press_key(self.action.pause_key())

    async def _execute_action(self, action: OperatorAction):
        """单 action 执行，包含完整的暂停/恢复外壳。"""
        import pydirectinput
        pause_key = self.action.pause_key()
        pydirectinput.keyDown(pause_key)
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)
        self.timer.pause()
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
                await asyncio.sleep(0.05)
                pydirectinput.keyUp(pause_key)
                self.timer.resume()
                await asyncio.sleep(0.05)

    async def _execute_batch(self, batch: List[OperatorAction]):
        """批量执行同 time_ms 的操作，只暂停/恢复一次游戏。"""

        import pydirectinput
        pause_key = self.action.pause_key()
        pydirectinput.keyDown(pause_key)
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)
        self.timer.pause()
        await asyncio.sleep(1.0)

        try:
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
            await asyncio.sleep(0.05)
            pydirectinput.keyUp(pause_key)
            self.timer.resume()
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
        await asyncio.sleep(0.05)
        pydirectinput.keyUp(pause_key)
        self.timer.pause()
        await asyncio.sleep(1.0)

        try:
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
                    self.timer.adjust(33.0)
                    print("__TIMER_ADJUST__:33.0")
                    if pos0:
                        self.action.select_at(pos0[0], pos0[1])
                        await asyncio.sleep(1.0)
        finally:
            await asyncio.sleep(1.0)
            pydirectinput.keyDown(pause_key)
            await asyncio.sleep(0.05)
            pydirectinput.keyUp(pause_key)
            self.timer.resume()
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
