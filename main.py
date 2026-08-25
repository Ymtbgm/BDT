import asyncio
import io
import json
import sys
import time
from typing import Optional

# 强制 stdout/stderr 使用 utf-8，避免 Windows GBK 编码下打印 emoji 等特殊字符崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 在 QProcess 子进程中，stderr 独立管道容易因输出量过大而阻塞子进程。
# 把 stderr 合并到 stdout，只保留一条管道，避免加载模型时卡顿。
if getattr(sys.stderr, "isatty", lambda: False)() is False:
    sys.stderr = sys.stdout

# QProcess 管道环境下默认是全缓冲，导致 print 不实时显示。
# 包装 stdout/stderr，每次 write 后自动 flush，确保日志即时输出。
class _AutoFlush:
    def __init__(self, f):
        self._f = f
    def write(self, s):
        r = self._f.write(s)
        self._f.flush()
        return r
    def flush(self):
        self._f.flush()
    def __getattr__(self, name):
        return getattr(self._f, name)

sys.stdout = _AutoFlush(sys.stdout)
sys.stderr = _AutoFlush(sys.stderr)
import cv2
import numpy as np
import keyboard
import pydirectinput

from core.capture.capture import WindowCapture
from core.vision.ocr_engine import OCREngine
from core.control.executor import ScriptExecutor
from core.vision.leak_detector import LeakDetector
from core.control.stage_selector import StageSelector
from core.control.retry_handler import StageRetryHandler
from core.game_state.cost_bar_start import CostBarStartDetector
from core.game_state.cost_bar_sync import CostBarSync
from core.game_state.cost_bar_sync_cc import CostBarSyncCC
from core.game_state.cost_bar_calibration import list_calibrations
from core.base.logging_utils import set_verbose
from core.base.paths import game_template, PROJECT_ROOT
from models.script_schema import ScriptModel
import action


class Runner:
    def __init__(self, debug: bool = False, cost_tag: Optional[str] = None, ocr_engine: Optional[str] = None):
        import time
        self.debug = debug
        self.cost_tag = cost_tag
        if self.debug:
            print(f"[后端] Runner.__init__ 开始，debug={debug}, cost_tag={cost_tag}")

        if self.debug:
            set_verbose(True)

        # 先注册热键，确保在后续耗时初始化（OCR 加载等）过程中 F12 也能被响应
        self._running = False
        self._abort = False
        self._stopping = False
        self._leak_detected = False
        self._leak_reason: Optional[str] = None
        self.avg_capture_ms = 0.0
        self._setup_hotkeys()

        try:
            self.capture = WindowCapture(backend="mss", debug=self.debug)

            # engine: None 表示自动选择（优先 PaddleX ONNX Runtime，回退 PaddleOCR）
            # model_size: "mobile" 模型体积小、速度快；"server" 精度高但慢
            t0 = time.perf_counter()
            self.ocr = OCREngine(
                use_gpu=False,
                debug=debug,
                engine=ocr_engine,
                model_size="mobile",
            )
            t1 = time.perf_counter()
            if self.debug:
                print(f"[DEBUG] OCREngine 初始化总耗时: {(t1 - t0) * 1000:.1f}ms")

            self.executor = ScriptExecutor(self.capture, self.ocr, action, debug=self.debug)
            self.executor.on_special_behavior_failed = lambda: self._on_leak(reason="probability_checkpoint")
            if cost_tag == "no_regen":
                print("[费用条同步] 费用不自然回复模式：禁用费用条同步，执行器纯按计时器驱动")
                self.cost_sync = None
            elif cost_tag:
                print(f"[费用条同步] 使用危机合约校准模式: {cost_tag}")
                self.cost_sync = CostBarSyncCC(self.capture, calibration_name=cost_tag, debug=self.debug)
            else:
                print("[费用条同步] 使用普通模式校准表（前 10s 区分 29 帧）")
                self.cost_sync = CostBarSyncCC(
                    self.capture,
                    calibration_name="normal",
                    calibration_schedule=[
                        (0.0, "normal_early"),
                        (10000.0, "normal"),
                    ],
                    debug=self.debug,
                )
            self.executor.set_cost_sync(self.cost_sync)
            self.leak = LeakDetector(self.capture)
            # max_side: 9999 表示不缩放，使用原图分辨率以获得最佳识别精度
            self.selector = StageSelector(self.capture, self.ocr, debug=debug, max_side=9999)

            # 资源路径兼容开发环境与 PyInstaller 打包环境
            _root = PROJECT_ROOT

            # 初始化漏怪重试处理器
            template_path = str(game_template("loss.png"))
            failed_template_path = str(game_template("failed.png"))
            mission_end_template_path = str(game_template("retry.png"))
            self.retry_handler = StageRetryHandler(
                self.capture,
                self.selector,
                template_path=template_path,
                failed_template_path=failed_template_path,
                mission_end_template_path=mission_end_template_path,
                debug=self.debug,
            )

            # 加载 COST 模板用于计时校准
            cost_path = CostBarStartDetector.default_template_path(_root)
            self.cost_template = CostBarStartDetector.load_template(str(cost_path))
            if self.cost_template is None:
                print(f"[警告] 无法加载 COST 模板: {cost_path}")

            # 加载行动结束模板用于无限凸图结算检测
            retry_path = game_template("retry.png")
            self.retry_template = cv2.imdecode(np.fromfile(str(retry_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            if self.retry_template is None:
                print(f"[警告] 无法加载 retry 模板: {retry_path}")
            else:
                if self.retry_template.ndim == 3 and self.retry_template.shape[2] == 3:
                    self.retry_template = cv2.cvtColor(self.retry_template, cv2.COLOR_BGR2BGRA)
        except Exception as e:
            # GUI 进程通过 QProcess 捕获该标记并弹窗提示，方便打包后无控制台时排查
            print(f"__INIT_ERROR__: {type(e).__name__}: {e}")
            raise

    async def _benchmark_capture_delay(self, samples: int = 5):
        """启动时 benchmark 截图延迟，用于动态修正帧补偿。"""
        times = []
        for _ in range(samples):
            t0 = time.perf_counter()
            try:
                _ = self.capture.capture()
            except Exception:
                pass
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            await asyncio.sleep(0.01)
        self.avg_capture_ms = sum(times) / len(times) if times else 0.0
        if self.debug:
            print(f"[计时校准] 截图延迟 benchmark: {self.avg_capture_ms:.2f}ms")
            print(f"[DEBUG] 单次截图耗时: {[f'{t:.2f}ms' for t in times]}")

    def _setup_hotkeys(self):
        keyboard.add_hotkey("f12", self._emergency_pause)
        keyboard.add_hotkey("f11", self._toggle_pause)

    def _emergency_pause(self):
        import time
        import pydirectinput

        now = time.time()
        if hasattr(self, '_last_emergency') and now - self._last_emergency < 1.0:
            return
        self._last_emergency = now
        print("[紧急暂停] 正在暂停游戏并停止脚本...")
        # 保持与 action.press_key 一致：按下 50ms 后松开，确保游戏能识别暂停键
        try:
            pause_key = action.pause_key()
            pydirectinput.keyDown(pause_key)
            time.sleep(0.05)
            pydirectinput.keyUp(pause_key)
        except Exception:
            pass
        try:
            self.executor.stop()
        except Exception:
            pass
        try:
            self.leak.stop()
        except Exception:
            pass
        self._running = False
        self._stopping = True
        self._abort = True

    def _toggle_pause(self):
        state = self.executor.get_state()
        pause_key = action.pause_key()
        if state.is_running:
            self.executor.pause()
            action.press_key(pause_key)
            print("[脚本暂停]")
        elif state.is_paused:
            action.press_key(pause_key)
            self.executor.resume()
            print("[脚本恢复]")

    def _on_leak(self, reason: str = "leak"):
        label = "漏怪" if reason == "leak" else "失败"
        print(f"[{label}检测] 检测到{label}，停止当前脚本...")
        self._leak_detected = True
        self._leak_reason = reason
        self.executor.stop()

    async def _wait_for_game_start(self, cost_threshold: float = 0.8, interval: float = 0.01, bar_timeout: float = 10.0) -> float:
        """完整流程：检测 COST → 等待 37 帧 → 检测费用条变化 → 返回截图半周期修正值(ms)。"""
        if self.cost_template is None:
            if self.debug:
                print("[计时校准] COST 模板未加载，跳过检测")
            return 0.0

        detector = CostBarStartDetector(
            self.capture,
            self.cost_template,
            debug=self.debug,
        )
        return await detector.detect_async(
            cost_threshold=cost_threshold,
            bar_timeout=bar_timeout,
            interval=interval,
            should_stop=lambda: self._stopping,
        )

    async def _monitor_leak_template(
        self,
        check_interval: float = 1.0,
        consecutive_required: int = 1,
    ):
        """使用模板匹配后台监控漏怪或失败提示。

        通过较高的匹配阈值（默认 0.9）避免技能特效/UI 闪烁导致的误触发，
        同时保留 consecutive_required 参数以便未来需要时开启连续帧确认。
        """
        consecutive_leak = 0
        consecutive_failed = 0
        while self._running and not self.executor._stop_event.is_set():
            try:
                is_leak = self.retry_handler.check_leak()
                is_failed = self.retry_handler.check_failed()

                if is_leak:
                    consecutive_leak += 1
                    consecutive_failed = 0
                    if self.debug:
                        print(
                            f"[漏怪监控] 漏怪疑似触发，连续 {consecutive_leak}/{consecutive_required} 次"
                        )
                    if consecutive_leak >= consecutive_required:
                        if self.debug:
                            print("[漏怪监控] 本轮检测: 漏怪触发")
                        self._on_leak(reason="leak")
                        return
                elif is_failed:
                    consecutive_leak = 0
                    consecutive_failed += 1
                    if self.debug:
                        print(
                            f"[漏怪监控] 失败疑似触发，连续 {consecutive_failed}/{consecutive_required} 次"
                        )
                    if consecutive_failed >= consecutive_required:
                        if self.debug:
                            print("[漏怪监控] 本轮检测: 失败触发")
                        self._on_leak(reason="failed")
                        return
                else:
                    if consecutive_leak > 0 or consecutive_failed > 0:
                        consecutive_leak = 0
                        consecutive_failed = 0
                        if self.debug:
                            print("[漏怪监控] 连续计数重置")
                    elif self.debug:
                        print("[漏怪监控] 本轮检测: 未触发")
            except Exception as e:
                print(f"[漏怪监控] 检测异常: {e}")
            await asyncio.sleep(check_interval)

    async def _wait_for_mission_end(
        self,
        check_interval: float = 0.5,
        timeout: float = 120.0,
        threshold: float = 0.8,
    ) -> bool:
        """等待关卡结束（行动结束界面出现），返回是否检测到。"""
        if self.retry_template is None:
            if self.debug:
                print("[结算检测] retry 模板未加载，跳过检测")
            return False

        # 统一模板通道格式为 BGR，避免与 ROI 通道不一致导致 OpenCV 异常
        template = self.retry_template
        if template.ndim == 3 and template.shape[2] == 4:
            template = cv2.cvtColor(template, cv2.COLOR_BGRA2BGR)
        elif template.ndim == 2:
            template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)

        if self.debug:
            print(f"[结算检测] 模板形状: {template.shape}, dtype={template.dtype}")

        # ROI 基于 2560x1600，按当前窗口缩放
        win_left = self.capture.monitor.get("left", 0)
        win_top = self.capture.monitor.get("top", 0)
        w, h = self.capture.get_window_size()
        x = win_left + int(w * 102 / 2560)
        y = win_top + int(h * 468 / 1600)
        roi_w = int(w * 534 / 2560)
        roi_h = int(h * 192 / 1600)

        start = time.perf_counter()
        while self._running and time.perf_counter() - start < timeout:
            try:
                roi = self.capture.capture_roi(x, y, roi_w, roi_h)
                if roi.size == 0:
                    await asyncio.sleep(check_interval)
                    continue
                # 统一 ROI 通道格式
                if roi.ndim == 3 and roi.shape[2] == 4:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
                elif roi.ndim == 2:
                    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

                if (
                    roi.shape[0] < template.shape[0]
                    or roi.shape[1] < template.shape[1]
                ):
                    if self.debug:
                        print(
                            f"[结算检测] ROI({roi.shape[1]}x{roi.shape[0]}) 小于模板"
                            f"({template.shape[1]}x{template.shape[0]})"
                        )
                    await asyncio.sleep(check_interval)
                    continue
                if self.debug:
                    print(f"[结算检测] ROI形状: {roi.shape}, dtype={roi.dtype}")
                result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if self.debug:
                    print(f"[结算检测] 匹配值={max_val:.3f}, 阈值={threshold}")
                if max_val >= threshold:
                    print(f"[结算检测] 检测到行动结束 (置信度: {max_val:.3f})")
                    return True
            except Exception as e:
                print(f"[结算检测] 检测异常: {e}")
                import traceback
                traceback.print_exc()
            await asyncio.sleep(check_interval)
        return False

    def _click_region_center(self, x: int, y: int, w: int, h: int):
        """点击指定区域的中心。"""
        import pydirectinput

        cx = x + w // 2
        cy = y + h // 2
        pydirectinput.moveTo(cx, cy)
        pydirectinput.click(button="left")

    async def run_script(
        self,
        script_path: str,
        loop_mode: bool = False,
        leak_mode: bool = False,
        auto_select_stage: bool = True,
        borrow_support: bool = False,
        support_friend_index: Optional[int] = None,
        support_skill: int = 1,
        support_module: int = 1,
        direct_start: bool = False,
        challenge_mode: bool = False,
        sand_table: bool = False,
        speed2x: bool = False,
    ):
        if self._abort:
            print("[紧急暂停] 初始化阶段已收到暂停指令，直接退出")
            return
        if challenge_mode and sand_table:
            print("[紧急暂停] 突袭模式与沙盘推演不能同时开启")
            return
        self._abort = False
        self._stopping = False
        with open(script_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        script = ScriptModel(**data)
        self.executor.load_script(script, borrow_support=borrow_support, direct_start=direct_start)

        print(f"脚本加载完成: {script.stage_code or '未命名'}")
        print(f"地图格子: {script.grid_rows}x{script.grid_cols}")
        print(f"操作数: {len(script.actions)}")

        # 自动选择关卡
        if auto_select_stage and script.stage_code:
            print(f"[自动选关] 尝试进入关卡: {script.stage_code}")
            ok = await self.selector.enter_stage(
                script.stage_code,
                borrow_support=borrow_support,
                support_friend_index=support_friend_index,
                support_skill=support_skill,
                support_module=support_module,
                direct_start=direct_start,
                challenge_mode=challenge_mode,
                sand_table=sand_table,
                should_stop=lambda: self._stopping,
            )
            if not ok:
                print("[自动选关] 失败，请手动进入关卡后按 Enter 继续...")
                if sys.stdin.isatty():
                    input()
            else:
                print("[系统] 已进入关卡，脚本准备就绪，按 Enter 开始执行...")
                if sys.stdin.isatty():
                    input()
        else:
            print("[系统] 脚本准备就绪，按 Enter 开始执行，F12 紧急暂停，F11 暂停/恢复...")
            if sys.stdin.isatty():
                input()

        if self._abort:
            print("[紧急暂停] 初始化阶段已收到暂停指令，直接退出")
            return

        self._running = True
        self._leak_retried = False
        print("[系统] 脚本开始运行")
        while self._running:
            self.executor._stop_event.clear()
            self._leak_detected = False
            self._leak_reason = None
            # 每次重新开始都重置 executor 状态（pool、grid 等）
            self.executor.load_script(script, borrow_support=borrow_support, direct_start=direct_start)

            # 计时器在进入关卡后启动（首次在循环外 enter_stage，重试在 handle_leak_once 后）
            offset_ms = await self._wait_for_game_start()
            if offset_ms > 0:
                if self.debug:
                    print(f"[计时校准] 费用条变化已检测，启动计时器并向前修正 {offset_ms:.1f}ms")
                self.executor.timer.start(offset_ms=offset_ms)
            else:
                if self.debug:
                    print("[计时校准] 检测失败，直接启动计时器")
                self.executor.timer.start()

            # 二倍数凸图：在费用条开始移动后，立即暂停，再在暂停状态下切 2 倍速并压缩未来操作时间
            if speed2x:
                # 先暂停游戏和计时器，避免在 1x 速度下继续跑时间
                pydirectinput.keyDown(action.pause_key())
                self.executor.timer.pause()
                await asyncio.sleep(0.05)
                pydirectinput.keyUp(action.pause_key())
                await asyncio.sleep(1.0)

                current_ms = self.executor.calibrate_timer_at_pause()
                speed_script = script.model_copy(deep=True)
                if self.debug:
                    print(f"[二倍数凸图] 当前时间 {current_ms:.1f}ms，开始压缩未来操作时间")
                for action_item in speed_script.actions:
                    if action_item.time_ms > current_ms:
                        original_time = action_item.time_ms
                        action_item.time_ms = int(current_ms + (action_item.time_ms - current_ms) / 2)
                        if self.debug:
                            print(f"  {action_item.action.value} {action_item.operator_name}: 原始={original_time}ms -> 压缩={action_item.time_ms}ms")
                speed_script.sort_actions()
                self.executor.load_script(speed_script, borrow_support=borrow_support, direct_start=direct_start)
                self.executor.set_speed2x_reference(current_ms)
                if self.debug:
                    print(f"[二倍数凸图] 已压缩未来操作时间，当前时间 {current_ms:.1f}ms")

                action.press_key(action.speed_key())
                await asyncio.sleep(0.5)

                pydirectinput.keyDown(action.pause_key())
                self.executor.timer.resume()
                await asyncio.sleep(0.05)
                pydirectinput.keyUp(action.pause_key())
                await asyncio.sleep(0.1)

            # 启动漏怪监控（模板匹配）
            leak_task = None
            if leak_mode:
                leak_task = asyncio.create_task(self._monitor_leak_template(check_interval=5.0))

            await self.executor.run()

            # 停止漏怪监控
            if leak_task is not None and not leak_task.done():
                leak_task.cancel()
                try:
                    await leak_task
                except asyncio.CancelledError:
                    pass

            if not self._running:
                break

            # 如果检测到漏怪、失败或概率点检查失败，执行对应重试流程
            if self._leak_detected:
                is_failed = self._leak_reason == "failed"
                is_probability = self._leak_reason == "probability_checkpoint"
                reason_text = "失败" if is_failed else ("概率点检查" if is_probability else "漏怪")

                # 概率点检查失败：总是无限重试，直到条件满足
                if is_probability:
                    print("[概率点检查] 条件不满足，重新开始关卡...")
                    ok = await self.retry_handler.handle_leak_once(
                        script.stage_code,
                        should_stop=lambda: self._stopping,
                        challenge_mode=challenge_mode,
                        sand_table=sand_table,
                    )
                    if not ok:
                        print("[概率点检查] 重试进入关卡失败，停止运行")
                        break
                    print("[概率点检查] 重试成功，重新开始执行...")
                    continue

                # 非无限凸图模式下只补打一次，再次触发则停止
                if not loop_mode and self._leak_retried:
                    print(f"[{reason_text}检测] 补打后再次{reason_text}，停止运行")
                    break
                label = "[无限凸图]" if loop_mode else f"[{reason_text}检测]"
                print(f"{label} 检测到{reason_text}，执行重试...")
                if is_failed:
                    ok = await self.retry_handler.handle_failed_once(
                        script.stage_code,
                        should_stop=lambda: self._stopping,
                        challenge_mode=challenge_mode,
                        sand_table=sand_table,
                    )
                else:
                    ok = await self.retry_handler.handle_leak_once(
                        script.stage_code,
                        should_stop=lambda: self._stopping,
                        challenge_mode=challenge_mode,
                        sand_table=sand_table,
                    )
                if not ok:
                    print(f"{label} 重试进入关卡失败，停止运行")
                    break
                if not loop_mode:
                    self._leak_retried = True
                print(f"{label} 重试成功，重新开始执行...")
                continue

            if not loop_mode:
                break

            print("[无限凸图] 脚本已执行完毕，等待关卡结束...")
            # ROI 基于 2560x1600，按当前窗口缩放
            win_left = self.capture.monitor.get("left", 0)
            win_top = self.capture.monitor.get("top", 0)
            w, h = self.capture.get_window_size()
            retry_x = win_left + int(w * 102 / 2560)
            retry_y = win_top + int(h * 468 / 1600)
            retry_w = int(w * 534 / 2560)
            retry_h = int(h * 192 / 1600)

            if await self._wait_for_mission_end():
                print("[无限凸图] 检测到行动结束，5 秒后点击继续...")
                await asyncio.sleep(5.0)
                if not self._running:
                    break
                self._click_region_center(retry_x, retry_y, retry_w, retry_h)
                await asyncio.sleep(3.0)
                if not self._running:
                    break
                print("[无限凸图] 重新选关并进入下一局...")
                ok = await self.selector.enter_stage(
                    script.stage_code,
                    borrow_support=borrow_support,
                    support_friend_index=support_friend_index,
                    support_skill=support_skill,
                    support_module=support_module,
                    direct_start=False,
                    challenge_mode=challenge_mode,
                    sand_table=sand_table,
                    should_stop=lambda: self._stopping,
                )
                if not ok:
                    print("[无限凸图] 重新选关失败，停止运行")
                    break
            else:
                print("[无限凸图] 未检测到行动结束，直接继续...")

            print("[无限凸图] 准备再次执行脚本...")
            continue

        print("执行结束")


async def main():
    if len(sys.argv) < 2:
        print("用法: python main.py <script.json> [--loop] [--leak] [--debug] ...")
        sys.exit(1)

    loop_mode = "--loop" in sys.argv
    leak_mode = "--leak" in sys.argv
    debug_mode = "--debug" in sys.argv
    borrow_support = "--borrow-support" in sys.argv
    direct_start = "--direct-start" in sys.argv
    challenge_mode = "--challenge-mode" in sys.argv
    sand_table = "--sand-table" in sys.argv
    speed2x = "--speed2x" in sys.argv
    if debug_mode:
        print(f"[后端] main() 启动，参数: {sys.argv}")
        print(f"[后端] debug={debug_mode}, loop={loop_mode}, leak={leak_mode}, challenge={challenge_mode}, sand_table={sand_table}")
    if challenge_mode and direct_start:
        print("错误：--challenge-mode（突袭模式）与 --direct-start（直接开始作战）不能同时开启")
        sys.exit(1)

    if loop_mode and direct_start:
        print("错误：--loop（无限凸图）与 --direct-start（直接开始作战）不能同时开启")
        sys.exit(1)

    if challenge_mode and sand_table:
        print("错误：--challenge-mode（突袭模式）与 --sand-table（沙盘推演）不能同时开启")
        sys.exit(1)

    def _arg_int(flag: str, default: int) -> int:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                try:
                    return int(sys.argv[idx + 1])
                except ValueError:
                    pass
        return default

    support_friend_index = None
    if "--support-friend-index" in sys.argv:
        idx = sys.argv.index("--support-friend-index")
        if idx + 1 < len(sys.argv):
            try:
                support_friend_index = int(sys.argv[idx + 1])
            except ValueError:
                pass

    support_skill = _arg_int("--support-skill", 1)
    support_module = _arg_int("--support-module", 1)

    def _arg_str(flag: str, default: str) -> str:
        if flag in sys.argv:
            idx = sys.argv.index(flag)
            if idx + 1 < len(sys.argv):
                return sys.argv[idx + 1]
        return default

    cost_tag = _arg_str("--cost-tag", None)

    if cost_tag and cost_tag not in list_calibrations() and cost_tag != "no_regen":
        print(f"错误：--cost-tag 必须是 {list_calibrations()} 之一，或 no_regen")
        sys.exit(1)

    ocr_engine_choice = _arg_str("--ocr-engine", "auto")
    ocr_engine = None if ocr_engine_choice == "auto" else ocr_engine_choice

    pause_key = _arg_str("--pause-key", "space")
    skill_key = _arg_str("--skill-key", "e")
    retreat_key = _arg_str("--retreat-key", "q")
    speed_key = _arg_str("--speed-key", "f")
    action.configure_keys(pause=pause_key, skill=skill_key, retreat=retreat_key, speed=speed_key)

    runner = Runner(debug=debug_mode, cost_tag=cost_tag, ocr_engine=ocr_engine)
    try:
        await runner.run_script(
            sys.argv[1],
            loop_mode=loop_mode,
            leak_mode=leak_mode,
            borrow_support=borrow_support,
            support_friend_index=support_friend_index,
            support_skill=support_skill,
            support_module=support_module,
            direct_start=direct_start,
            challenge_mode=challenge_mode,
            sand_table=sand_table,
            speed2x=speed2x,
        )
    finally:
        # 显式清理本进程注册的全局键盘钩子，避免退出后残留影响 GUI 进程
        try:
            keyboard.unhook_all()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"[后端] 未捕获异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
