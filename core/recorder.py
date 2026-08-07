import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pynput import mouse, keyboard

import action
import core.constants as constants
from core.avatar_matcher import AvatarMatcherBase, create_avatar_matcher
from core.capture import WindowCapture
from core.ocr_engine import OCREngine
from core.region_state_timer import RegionStateTimer
from core.tile_pos import TilePosCalculator, load_stage_dimensions
from models.raw_recording import RawRecording, RawAction, Keyframe, KeyframeType
from models.script_schema import ScriptModel, ActionType


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

    def __init__(
        self,
        capture: WindowCapture,
        timer: Optional[RegionStateTimer] = None,
        stage_code: str = "",
        debug: bool = False,
        debug_cost_bar: bool = False,
        debug_resolver: bool = False,
        debug_screenshot: bool = False,
        initial_operator_count: int = 0,
        initial_item_count: int = 0,
        support_count: int = 0,
        pause_key: str = "p",
        matchstick_hotkeys: Optional[dict] = None,
        cost_bar_calibration_name: Optional[str] = None,
        ocr: Optional[OCREngine] = None,
        avatar_model_name: str = "resnet18",
        resolver_log_callback: Optional[Callable[[str], None]] = None,
    ):
        self.capture = capture
        self.timer = timer
        # debug: 录制器自身状态机/原始操作日志
        # debug_cost_bar: RegionStateTimer 费用条检测日志
        # debug_resolver: OfflineResolver 离线识别日志
        # debug_screenshot: DEPLOY/RETREAT/SKILL 调试截图
        self.debug = debug
        self.debug_cost_bar = debug_cost_bar
        self.debug_resolver = debug_resolver
        self.debug_screenshot = debug_screenshot
        self._pause_key = pause_key
        self._matchstick_hotkeys = matchstick_hotkeys
        self._cost_bar_calibration_name = cost_bar_calibration_name
        self.avatar_model_name = avatar_model_name
        self.resolver_log_callback = resolver_log_callback

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

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------
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
        self._reset_state()

        self._state = "WAITING_FOR_START"
        self._wait_thread = threading.Thread(target=self._wait_for_timer_start, daemon=True)
        self._wait_thread.start()
        self._log("start() 进入 WAITING_FOR_START，等待 cost 检测...")

        self._mouse_listener = mouse.Listener(
            on_click=self._on_click,
            on_move=self._on_move,
        )
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_press,
        )
        self._mouse_listener.start()
        self._keyboard_listener.start()
        self._log("监听器已启动")

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
        self._log(f"停止录制，共录制 {len(self._raw_actions)} 个原始操作")
        return self._resolve_recording()

    def is_recording(self) -> bool:
        return self._recording

    def is_stop_requested(self) -> bool:
        return self._stop_requested

    def is_squad_capture_done(self) -> bool:
        return getattr(self, "_squad_keyframes_captured", False)

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

        captured = 0
        for roi_index in range(len(name_ratios)):
            if not has_support and roi_index == 12:
                # 不携带助战时，跳过第 13 个槽位（右上角助战位）
                continue
            if captured >= self.initial_operator_count:
                break
            if roi_index >= len(avatar_ratios):
                break

            rx, ry, rw, rh = name_ratios[roi_index]
            x = left + int(w * rx)
            y = top + int(h * ry)
            roi_w = int(w * rw)
            roi_h = int(h * rh)
            try:
                name_img = self.capture.capture_roi(x, y, roi_w, roi_h)
            except Exception:
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
                self._log(f"编队槽位 {captured} 名称识别失败，使用占位符")
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
            except Exception:
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

            self._log(f"编队槽位 {captured} (ROI={roi_index}): {name} (conf={best_conf:.2f})")
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
        from core.resolver import OfflineResolver
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

    def _nearest_grid(self, win_x: int, win_y: int, side: bool = False) -> Optional[Tuple[int, int]]:
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

    def _bar_positions(self) -> Dict[int, Tuple[int, int]]:
        """返回部署栏索引（0 为最右侧）到绝对屏幕坐标的映射。"""
        total = self._total_bar_slots()
        if total == 0:
            return {}
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        bar_y = int(h * 1500 / 1600)
        cell_w = w / 12 if total <= 12 else w / total
        positions = {}
        for i in range(total):
            cx = w - cell_w * (i + 0.5)
            positions[i] = (left + int(cx), top + bar_y)
        return positions

    def _bar_index_at(self, win_x: int, win_y: int) -> Optional[int]:
        total = self._total_bar_slots()
        if total == 0:
            return None
        positions = self._bar_positions()
        cell_w = self.capture.get_window_size()[0] / 12 if total <= 12 else self.capture.get_window_size()[0] / total
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
        被误判为已移出部署区。
        """
        total = self._total_bar_slots()
        if total == 0:
            return False
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)

        # y 范围与 _bar_capture_roi 保持一致：上移到 1370，覆盖到窗口底部
        bar_top = top + int(h * self._BAR_CAPTURE_TOP_RATIO) - 20
        bar_bottom = top + h

        # x 范围：实际有 slot 的区域，从右侧向左 total * cell_w
        cell_w = w / 12 if total <= 12 else w / total
        bar_left = left + max(0, int(w - cell_w * total))
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
        except Exception:
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
        self._log(f"_reset_state 从 {self._state} 重置为 IDLE")
        with self._lock:
            self._state = "IDLE"
            self._pending = None
            self._selected_unit_grid = None
            self._cancel_timeout()

    def _wait_for_timer_start(self):
        self._log("_wait_for_timer_start 开始")

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
                bar_idx = self._bar_index_at(win_x, win_y)
                if bar_idx is not None:
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
                                      self._RETREAT_X, self._RETREAT_Y,
                                      self._RETREAT_W, self._RETREAT_H)
                in_skill = self._in_fixed_roi(win_x, win_y,
                                      self._SKILL_X, self._SKILL_Y,
                                      self._SKILL_W, self._SKILL_H)
                self._log(f" UNIT_SELECTED mouseUp in_retreat={in_retreat} in_skill={in_skill}")
                if in_retreat:
                    self._record_raw_retreat(self._selected_unit_grid, int(self._now_ms()))
                    self._reset_state()
                    return
                if in_skill:
                    self._record_raw_skill(self._selected_unit_grid, int(self._now_ms()))
                    self._reset_state()
                    return
                self._log("UNIT_SELECTED 点击空地，丢弃")
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
        if key == keyboard.Key.f10:
            self._log("F10 停止录制")
            self._stop_requested = True
            return
        char = getattr(key, "char", None)
        with self._lock:
            state = self._state

        if state == "WAITING_FOR_START":
            return

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
                self._log(f" {action.skill_key().upper()}键技能 {self._selected_unit_grid}")
                self._cancel_timeout()
                self._record_raw_skill(self._selected_unit_grid, int(self._now_ms()))
                self._reset_state()

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
