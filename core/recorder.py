import threading
import time
from typing import List, Optional, Tuple, Dict

from pynput import mouse, keyboard

import action
from core.capture import WindowCapture
from core.region_state_timer import RegionStateTimer
from core.operator_pool import OperatorPool
from core.tile_pos import TilePosCalculator
from models.script_schema import ScriptModel, OperatorAction, ActionType, ItemInfo


class ActionRecorder:
    """基于全局输入监听的操作录制器。

    使用 pynput 监听键盘/鼠标，通过状态机识别 DEPLOY、RETREAT、SKILL 操作。
    时间基准完全委托给 RegionStateTimer，录制器自身不维护计时逻辑。
    """

    # 基于 2560x1600 的固定 ROI（选中干员后视角居中，按钮位置固定）
    _RETREAT_X = 1145
    _RETREAT_Y = 510
    _RETREAT_W = 170
    _RETREAT_H = 160
    _SKILL_X = 1615
    _SKILL_Y = 885
    _SKILL_W = 250
    _SKILL_H = 200

    _BASE_W = 2560
    _BASE_H = 1600

    # 拖拽判定：mouseUp 与 mouseDown 距离超过此阈值视为拖拽（而非点击）
    _DRAG_THRESHOLD = 20
    # 方向选择：二次拖拽距离超过此阈值才判定有方向
    _DIR_THRESHOLD = 20
    # 状态机超时（秒）
    _TIMEOUT_DEPLOY_DIR = 2.0
    _TIMEOUT_UNIT_SELECT = 2.0

    def __init__(
        self,
        capture: WindowCapture,
        timer: Optional[RegionStateTimer] = None,
        operators: Optional[List[str]] = None,
        items: Optional[List[ItemInfo]] = None,
        grid_rows: int = 7,
        grid_cols: int = 9,
        stage_code: Optional[str] = None,
        stage_name: Optional[str] = None,
        debug: bool = False,
    ):
        self.capture = capture
        self.timer = timer
        self.debug = debug

        self.operators = list(operators) if operators else []
        self.items = list(items) if items else []
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.stage_code = stage_code
        self.stage_name = stage_name

        w, h = capture.get_window_size()
        left = capture.monitor.get("left", 0)
        top = capture.monitor.get("top", 0)

        self.pool = OperatorPool(w, h, self.operators, self.items)
        self.pool.set_window_offset(left, top)

        self.tile_calc = TilePosCalculator(
            w, h, grid_rows, grid_cols,
            stage_code=stage_code, stage_name=stage_name,
        )

        self._scale_x = w / self._BASE_W
        self._scale_y = h / self._BASE_H

        # 录制状态
        self._recording = False
        self._actions: List[OperatorAction] = []
        self._deployed: Dict[str, Tuple[int, int]] = {}
        self._lock = threading.Lock()
        self._stop_requested = False
        self._wait_thread: Optional[threading.Thread] = None

        # 状态机
        self._state = "IDLE"
        self._pending: Optional[Dict] = None
        self._mouse_down_pos: Optional[Tuple[int, int]] = None
        self._mouse_down_time: Optional[float] = None
        self._selected_unit_pos: Optional[Tuple[int, int]] = None
        self._selected_unit_grid: Optional[Tuple[int, int]] = None
        self._timeout_timer: Optional[threading.Timer] = None

        # pynput 监听器
        self._mouse_listener: Optional[mouse.Listener] = None
        self._keyboard_listener: Optional[keyboard.Listener] = None

        # 时间基准完全委托给 RegionStateTimer
        # 若外部未提供计时器，录制器自行创建一个
        if self.timer is None:
            self.timer = RegionStateTimer(self.capture, debug=self.debug)
            self._own_timer = True
        else:
            self._own_timer = False
        self._get_time_ms = self.timer.get_elapsed_ms

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
    def start(self):
        if self._recording:
            if self.debug:
                print("[录制器] start() 被调用但已在录制中")
            return
        self._recording = True
        self._stop_requested = False
        self._actions.clear()
        self._deployed.clear()
        self._pending = None

        # 统一进入 WAITING_FOR_START，由 _wait_for_timer_start 执行 cost 检测或同步外部计时器
        self._state = "WAITING_FOR_START"
        self._wait_thread = threading.Thread(target=self._wait_for_timer_start, daemon=True)
        self._wait_thread.start()
        if self.debug:
            print("[录制器] start() 进入 WAITING_FOR_START，等待 cost 检测...")

        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_move=self._on_move,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()
        if self.debug:
            print("[录制器] 监听器已启动")

    def stop(self) -> ScriptModel:
        if not self._recording:
            if self.debug:
                print("[录制器] stop() 被调用但不在录制中")
            return self._build_script()
        self._recording = False
        self._stop_requested = True
        if self._mouse_listener:
            self._mouse_listener.stop()
        if self._keyboard_listener:
            self._keyboard_listener.stop()
        self._cancel_timeout()
        if self._wait_thread is not None:
            self._wait_thread.join(timeout=0.5)
        # 若计时器由录制器自行创建，一并停止
        if self._own_timer and self.timer is not None:
            self.timer.stop()
        if self.debug:
            print(f"[录制器] 停止录制，共录制 {len(self._actions)} 个操作")
            for i, a in enumerate(self._actions):
                print(f"  [{i}] {a.action} {a.operator_name} @ {a.grid} t={a.time_ms}")
        return self._build_script()

    def is_recording(self) -> bool:
        return self._recording

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    # ------------------------------------------------------------------
    # 输出构建
    # ------------------------------------------------------------------
    def _build_script(self) -> ScriptModel:
        script = ScriptModel(
            stage_code=self.stage_code,
            stage_name=self.stage_name,
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            operators=self.operators,
            items=[ItemInfo(name=i.name, charges=i.charges) for i in self.items],
            actions=[a.model_copy() for a in self._actions],
        )
        script.sort_actions()
        return script

    # ------------------------------------------------------------------
    # 坐标工具
    # ------------------------------------------------------------------
    def _win_xy(self, abs_x: int, abs_y: int) -> Tuple[int, int]:
        """屏幕绝对坐标 → 窗口相对坐标。"""
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return abs_x - left, abs_y - top

    def _nearest_grid(self, win_x: int, win_y: int, side: bool = False) -> Optional[Tuple[int, int]]:
        """找距离 (win_x, win_y) 最近的地图格子。"""
        best = None
        best_dist = float("inf")
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                sx, sy = self.tile_calc.get_screen_pos(r, c, side=side)
                d = (sx - win_x) ** 2 + (sy - win_y) ** 2
                if d < best_dist:
                    best_dist = d
                    best = (r, c)
        # 距离过远则视为不在场上
        if best_dist > (150 * min(self._scale_x, self._scale_y)) ** 2:
            return None
        return best

    def _bar_index_at(self, win_x: int, win_y: int) -> Optional[int]:
        """判断点击是否落在干员栏/道具栏某个头像内，返回索引或 None。"""
        total = len(self.pool.operators) + len(self.pool.items)
        if total == 0:
            return None
        cell_w = self.pool.window_width / 12 if total <= 12 else self.pool.window_width / total
        half = cell_w / 2
        for i, (cx, cy) in self.pool._bar_positions.items():
            # _bar_positions 返回绝对坐标，需转换
            rel_cx = cx - self.capture.monitor.get("left", 0)
            rel_cy = cy - self.capture.monitor.get("top", 0)
            if rel_cx - half <= win_x <= rel_cx + half and rel_cy - half <= win_y <= rel_cy + half:
                if self.debug:
                    print(f"[录制器] _bar_index_at hit idx={i} name={self._name_from_bar_index(i)} pos=({rel_cx:.0f},{rel_cy:.0f}) half={half:.1f}")
                return i
        if self.debug:
            print(f"[录制器] _bar_index_at miss pos=({win_x:.0f},{win_y:.0f}) bar_positions={self.pool._bar_positions}")
        return None

    def _name_from_bar_index(self, index: int) -> Optional[str]:
        """根据部署栏索引返回干员/道具名称。"""
        total_items = len(self.pool.items)
        # OperatorPool 内部：索引 0 是最右侧（道具优先）
        # _available_items 是从右到左排列的道具名
        if index < total_items:
            # 道具区域
            rev_idx = index
            if rev_idx < len(self.pool._available_items):
                return self.pool._available_items[rev_idx]
            return None
        else:
            # 干员区域
            op_idx_in_bar = index - total_items
            bar_indices = list(self.pool._bar_indices)
            if op_idx_in_bar < len(bar_indices):
                op_list_idx = bar_indices[-(op_idx_in_bar + 1)]
                return self.pool.operators[op_list_idx]
            return None

    def _is_item(self, name: str) -> bool:
        return name in {it.name for it in self.pool.items}

    def _in_roi(self, win_x: int, win_y: int, base_x: int, base_y: int,
                dx: int, dy: int, w: int, h: int) -> bool:
        x1 = base_x + dx * self._scale_x
        y1 = base_y + dy * self._scale_y
        x2 = x1 + w * self._scale_x
        y2 = y1 + h * self._scale_y
        return x1 <= win_x <= x2 and y1 <= win_y <= y2

    def _in_fixed_roi(self, win_x: int, win_y: int, base_x: int, base_y: int,
                      w: int, h: int) -> bool:
        """检查坐标是否落在固定 ROI 内（基于 2560x1600 的绝对坐标，按窗口缩放）。"""
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
        if self.debug:
            print(f"[录制器] _reset_state 从 {self._state} 重置为 IDLE")
        with self._lock:
            self._state = "IDLE"
            self._pending = None
            self._selected_unit_pos = None
            self._selected_unit_grid = None
            self._cancel_timeout()

    def _wait_for_timer_start(self):
        if self.debug:
            print("[录制器] _wait_for_timer_start 开始")

        # 确保计时器已以 cost 检测模式启动
        if not self.timer.is_running():
            if self.debug:
                print("[录制器] 启动计时器 (use_cost_detection=True)")
            self.timer.start(use_cost_detection=True)

        # 等待 cost 检测完成、计时器启动
        while not self._stop_requested:
            info = self.timer.tick()
            if info.get("started"):
                with self._lock:
                    self._state = "IDLE"
                if self.debug:
                    print(f"[录制器] 计时器已启动 (elapsed={info['elapsed_ms']:.1f}ms)，开始录制")
                break
            time.sleep(self.timer.frame_ms / 1000.0)

        # 计时器启动后，继续定期 tick()，确保 _update_time 的调用间隔
        # 不超过 frame_ms，从而避免 max_delta_ms clamp 导致的时间偏慢。
        while not self._stop_requested:
            self.timer.tick()
            time.sleep(self.timer.frame_ms / 1000.0)

        if self.debug:
            print("[录制器] _wait_for_timer_start 结束")

    def _now_ms(self) -> float:
        t = self._get_time_ms()
        if self.debug:
            print(f"[录制器] _now_ms={t:.1f}ms")
        return t

    # ------------------------------------------------------------------
    # 操作记录
    # ------------------------------------------------------------------
    def _record_deploy(self, name: str, grid: Tuple[int, int], direction: Optional[str], time_ms: Optional[int] = None):
        is_item = self._is_item(name)
        t = time_ms if time_ms is not None else int(self._now_ms())
        act = OperatorAction(
            time_ms=t,
            action=ActionType.DEPLOY,
            operator_name=name,
            grid=grid,
            direction=direction,
            is_object=False,
        )
        with self._lock:
            self._actions.append(act)
            if not is_item:
                self._deployed[name] = grid
            # 同步更新 pool 状态（从部署栏移除）
            self.pool.deploy(name, grid)
        if self.debug:
            print(f"[录制器] DEPLOY {name} @ {grid} dir={direction} time_ms={t} actions_count={len(self._actions)}")
            print(f"[录制器]  pool bar_indices={self.pool._bar_indices} items={self.pool._available_items}")

    def _record_retreat(self, name: str):
        act = OperatorAction(
            time_ms=int(self._now_ms()),
            action=ActionType.RETREAT,
            operator_name=name,
        )
        with self._lock:
            self._actions.append(act)
            if name in self._deployed:
                del self._deployed[name]
            self.pool.retreat(name)
        if self.debug:
            print(f"[录制器] RETREAT {name}")

    def _record_skill(self, name: str):
        act = OperatorAction(
            time_ms=int(self._now_ms()),
            action=ActionType.SKILL,
            operator_name=name,
        )
        with self._lock:
            self._actions.append(act)
        if self.debug:
            print(f"[录制器] SKILL {name}")

    # ------------------------------------------------------------------
    # 输入回调
    # ------------------------------------------------------------------
    def _on_click(self, abs_x, abs_y, button, pressed):
        if not self._recording:
            return
        win_x, win_y = self._win_xy(abs_x, abs_y)

        with self._lock:
            state = self._state

        if state == "WAITING_FOR_START":
            return

        if self.debug:
            print(f"[录制器] mouse{'Down' if pressed else 'Up'} state={state} pos=({win_x:.0f},{win_y:.0f})")

        if pressed:
            # mouseDown
            self._mouse_down_pos = (win_x, win_y)
            self._mouse_down_time = time.perf_counter()

            if state == "IDLE":
                bar_idx = self._bar_index_at(win_x, win_y)
                if self.debug:
                    print(f"[录制器]  IDLE mouseDown bar_idx={bar_idx}")
                if bar_idx is not None:
                    name = self._name_from_bar_index(bar_idx)
                    if self.debug:
                        print(f"[录制器]  IDLE mouseDown name={name}")
                    if name is not None:
                        with self._lock:
                            self._state = "DRAGGING"
                            self._pending = {
                                "type": "DEPLOY",
                                "name": name,
                                "start_pos": (win_x, win_y),
                            }
                        if self.debug:
                            print(f"[录制器] 开始拖拽: {name}")

            elif state == "AWAITING_DIRECTION":
                with self._lock:
                    self._pending["dir_down_pos"] = (win_x, win_y)
                if self.debug:
                    print(f"[录制器]  AWAITING_DIRECTION mouseDown dir_down=({win_x:.0f},{win_y:.0f})")

        else:
            # mouseUp
            if state == "DRAGGING":
                # 若释放位置仍在部署栏，视为未拖到场上，直接取消
                if self._bar_index_at(win_x, win_y) is not None:
                    if self.debug:
                        print("[录制器] 拖拽释放位置仍在部署栏，取消")
                    self._reset_state()
                    return
                # 部署时游戏为 side 视角，需用 side=True 计算最近格子
                grid = self._nearest_grid(win_x, win_y, side=True)
                if self.debug:
                    print(f"[录制器]  DRAGGING mouseUp nearest_grid={grid} pos=({win_x:.0f},{win_y:.0f})")
                if grid is not None:
                    with self._lock:
                        self._state = "AWAITING_DIRECTION"
                        self._pending["grid"] = grid
                        self._pending["time_ms"] = int(self._now_ms())
                    # 启动方向选择超时
                    self._set_timeout(self._TIMEOUT_DEPLOY_DIR, self._on_deploy_timeout)
                    if self.debug:
                        print(f"[录制器] 等待方向选择 @ {grid}")
                else:
                    if self.debug:
                        print("[录制器] 拖拽未落在场上，取消")
                    self._reset_state()

            elif state == "AWAITING_DIRECTION":
                self._cancel_timeout()
                if self._pending is None:
                    if self.debug:
                        print("[录制器] 错误: AWAITING_DIRECTION 但 _pending 为 None")
                    self._reset_state()
                    return
                dir_pos = self._pending.get("dir_down_pos")
                grid = self._pending["grid"]
                name = self._pending["name"]
                direction = None
                if dir_pos is not None:
                    dx = win_x - dir_pos[0]
                    dy = win_y - dir_pos[1]
                    dist = (dx ** 2 + dy ** 2) ** 0.5
                    if dist > self._DIR_THRESHOLD:
                        # 判断方向：取绝对值较大的分量
                        if abs(dx) > abs(dy):
                            direction = "right" if dx > 0 else "left"
                        else:
                            direction = "down" if dy > 0 else "up"
                if self.debug:
                    print(f"[录制器]  AWAITING_DIRECTION mouseUp name={name} grid={grid} dir={direction} dir_pos={dir_pos} win=({win_x:.0f},{win_y:.0f})")
                self._record_deploy(name, grid, direction)
                self._reset_state()

            elif state == "UNIT_SELECTED":
                # 选中干员后视角居中，撤退/技能按钮位置固定
                in_retreat = self._in_fixed_roi(win_x, win_y,
                                      self._RETREAT_X, self._RETREAT_Y,
                                      self._RETREAT_W, self._RETREAT_H)
                in_skill = self._in_fixed_roi(win_x, win_y,
                                      self._SKILL_X, self._SKILL_Y,
                                      self._SKILL_W, self._SKILL_H)
                if self.debug:
                    print(f"[录制器]  UNIT_SELECTED mouseUp in_retreat={in_retreat} in_skill={in_skill} pos=({win_x:.0f},{win_y:.0f})")
                if in_retreat:
                    name = self._deployed_name_at(self._selected_unit_grid)
                    if name:
                        self._record_retreat(name)
                    self._reset_state()
                    return
                if in_skill:
                    name = self._deployed_name_at(self._selected_unit_grid)
                    if name:
                        self._record_skill(name)
                    self._reset_state()
                    return
                # 点击空地 → 丢弃
                if self.debug:
                    print("[录制器] UNIT_SELECTED 点击空地，丢弃")
                self._reset_state()

            elif state == "IDLE":
                # 场上点击 → 选中干员（使用矩形区域命中，避免单像素误差）
                hit = self._deployed_hit(win_x, win_y)
                if hit is not None:
                    name, grid = hit
                    side_pos = self.tile_calc.get_screen_pos(*grid, side=True)
                    with self._lock:
                        self._state = "UNIT_SELECTED"
                        self._selected_unit_pos = side_pos
                        self._selected_unit_grid = grid
                    if self.debug:
                        print(f"[录制器] 选中干员 {name} @ {grid} (区域命中)")
                else:
                    if self.debug:
                        print(f"[录制器]  IDLE mouseUp 无部署干员 pos=({win_x:.0f},{win_y:.0f})")

    def _on_move(self, abs_x, abs_y):
        # 当前不需要追踪拖拽轨迹，只需起点和终点
        pass

    def _on_press(self, key):
        if not self._recording:
            return
        # F10 停止录制
        if key == keyboard.Key.f10:
            if self.debug:
                print("[录制器] F10 停止录制")
            self._stop_requested = True
            return
        char = getattr(key, "char", None)
        with self._lock:
            state = self._state

        if state == "WAITING_FOR_START":
            return

        if self.debug:
            print(f"[录制器] key_press char={char} key={key} state={state}")

        if state == "UNIT_SELECTED":
            name = self._deployed_name_at(self._selected_unit_grid)
            if name is None:
                if self.debug:
                    print(f"[录制器]  UNIT_SELECTED key_press 无干员 @ {self._selected_unit_grid}")
                return
            if char and char.lower() == action.retreat_key():
                if self.debug:
                    print(f"[录制器]  {action.retreat_key().upper()}键撤退 {name}")
                self._cancel_timeout()
                self._record_retreat(name)
                self._reset_state()
            elif char and char.lower() == action.skill_key():
                if self.debug:
                    print(f"[录制器]  {action.skill_key().upper()}键技能 {name}")
                self._cancel_timeout()
                self._record_skill(name)
                self._reset_state()

    # ------------------------------------------------------------------
    # 超时回调
    # ------------------------------------------------------------------
    def _on_deploy_timeout(self):
        with self._lock:
            if self._state != "AWAITING_DIRECTION":
                if self.debug:
                    print(f"[录制器] 方向选择超时回调被忽略，当前 state={self._state}")
                return
            pending = self._pending
        if pending and pending.get("type") == "DEPLOY":
            if self.debug:
                print(f"[录制器] 方向选择超时 name={pending['name']} grid={pending['grid']}")
            self._record_deploy(pending["name"], pending["grid"], None, time_ms=pending.get("time_ms"))
        else:
            if self.debug:
                print(f"[录制器] 方向选择超时但 pending 无效: {pending}")
        self._reset_state()
        if self.debug:
            print("[录制器] 方向选择超时，已重置状态")

    # ------------------------------------------------------------------
    # 辅助查询
    # ------------------------------------------------------------------
    def _deployed_hit(self, win_x: int, win_y: int) -> Optional[Tuple[str, Tuple[int, int]]]:
        """以点击位置为中心，在相邻格子距离为半径的矩形内查找已部署干员。

        返回 (name, grid) 或 None。
        """
        if not self._deployed:
            if self.debug:
                print(f"[录制器] _deployed_hit 无已部署干员")
            return None

        # 非 side 视角下相邻格子中心距离作为判定半径
        p00 = self.tile_calc.get_screen_pos(0, 0)
        p01 = self.tile_calc.get_screen_pos(0, 1)
        p10 = self.tile_calc.get_screen_pos(1, 0)
        dx = ((p01[0] - p00[0]) ** 2 + (p01[1] - p00[1]) ** 2) ** 0.5
        dy = ((p10[0] - p00[0]) ** 2 + (p10[1] - p00[1]) ** 2) ** 0.5
        radius = min(dx, dy)

        best_name = None
        best_grid = None
        best_dist = float("inf")
        with self._lock:
            for name, grid in self._deployed.items():
                sx, sy = self.tile_calc.get_screen_pos(*grid)
                dist = ((sx - win_x) ** 2 + (sy - win_y) ** 2) ** 0.5
                if self.debug:
                    print(f"[录制器] _deployed_hit {name}@{grid} screen=({sx},{sy}) dist={dist:.1f} click=({win_x},{win_y})")
                if dist < best_dist:
                    best_dist = dist
                    best_name = name
                    best_grid = grid

        if self.debug:
            print(f"[录制器] _deployed_hit best={best_name}@{best_grid} dist={best_dist:.1f} radius={radius:.1f} hit={'Y' if best_name and best_dist <= radius else 'N'}")

        if best_name is not None and best_dist <= radius:
            return best_name, best_grid
        return None

    def _deployed_name_at(self, grid: Optional[Tuple[int, int]]) -> Optional[str]:
        if grid is None:
            return None
        with self._lock:
            for name, g in self._deployed.items():
                if g == grid:
                    return name
        return None
