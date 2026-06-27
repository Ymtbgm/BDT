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
from pathlib import Path

from core.capture import WindowCapture
from core.ocr_engine import OCREngine
from core.executor import ScriptExecutor
from core.leak_detector import LeakDetector
from core.stage_selector import StageSelector
from core.retry_handler import StageRetryHandler
from core.cost_bar_start import CostBarStartDetector
from core.cost_bar_sync import CostBarSync
from core.cost_bar_sync_cc import CostBarSyncCC
from core.cost_bar_calibration import list_calibrations
from models.script_schema import ScriptModel
import action


class Runner:
    def __init__(self, debug: bool = False, cost_tag: Optional[str] = None):
        import time
        self.debug = debug
        self.cost_tag = cost_tag

        # 先注册热键，确保在后续耗时初始化（OCR 加载等）过程中 F12 也能被响应
        self._running = False
        self._abort = False
        self._stopping = False
        self._leak_detected = False
        self.avg_capture_ms = 0.0
        self._setup_hotkeys()

        self.capture = WindowCapture(backend="mss")

        # engine: None 表示让 PaddleOCR 自行选择（默认 paddle_static，性能最好）
        # 如果当前环境 paddle_static 崩溃，会自动回退到 transformers
        # model_size: "mobile" 模型体积小、速度快；"server" 精度高但慢
        t0 = time.perf_counter()
        self.ocr = OCREngine(
            use_gpu=False,
            debug=debug,
            engine=None,
            model_size="mobile",
        )
        t1 = time.perf_counter()
        if self.debug:
            print(f"[DEBUG] OCREngine 初始化总耗时: {(t1 - t0) * 1000:.1f}ms")

        self.executor = ScriptExecutor(self.capture, self.ocr, action, debug=self.debug)
        if cost_tag:
            print(f"[费用条同步] 使用危机合约校准模式: {cost_tag}")
            self.cost_sync = CostBarSyncCC(self.capture, calibration_name=cost_tag, debug=self.debug)
        else:
            self.cost_sync = CostBarSync(self.capture, debug=self.debug)
        self.executor.set_cost_sync(self.cost_sync)
        self.leak = LeakDetector(self.capture)
        # max_side: 9999 表示不缩放，使用原图分辨率以获得最佳识别精度
        self.selector = StageSelector(self.capture, self.ocr, debug=debug, max_side=9999)

        # 资源路径兼容开发环境与 PyInstaller 打包环境
        _root = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).parent

        # 初始化漏怪重试处理器
        template_path = str(_root / "core" / "resource" / "loss.png")
        self.retry_handler = StageRetryHandler(
            self.capture, self.selector, template_path=template_path, debug=self.debug
        )

        # 加载 COST 模板用于计时校准
        cost_path = CostBarStartDetector.default_template_path(_root)
        self.cost_template = CostBarStartDetector.load_template(str(cost_path))
        if self.cost_template is None:
            print(f"[警告] 无法加载 COST 模板: {cost_path}")

        # 加载行动结束模板用于无限凸图结算检测
        retry_path = _root / "core" / "resource" / "retry.png"
        self.retry_template = cv2.imdecode(np.fromfile(str(retry_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if self.retry_template is None:
            print(f"[警告] 无法加载 retry 模板: {retry_path}")
        else:
            if self.retry_template.ndim == 3 and self.retry_template.shape[2] == 3:
                self.retry_template = cv2.cvtColor(self.retry_template, cv2.COLOR_BGR2BGRA)

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

    def _on_leak(self):
        print("[漏怪检测] 检测到漏怪，停止当前脚本...")
        self._leak_detected = True
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

    async def _monitor_leak_template(self, check_interval: float = 1.0):
        """使用模板匹配后台监控漏怪。"""
        while self._running and not self.executor._stop_event.is_set():
            try:
                is_leak = self.retry_handler.check_leak()
                if self.debug:
                    print(f"[漏怪监控] 本轮检测: {'触发' if is_leak else '未触发'}")
                if is_leak:
                    self._on_leak()
                    return
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
                if (
                    roi.shape[0] < self.retry_template.shape[0]
                    or roi.shape[1] < self.retry_template.shape[1]
                ):
                    if self.debug:
                        print(
                            f"[结算检测] ROI({roi_w}x{roi_h}) 小于模板"
                            f"({self.retry_template.shape[1]}x{self.retry_template.shape[0]})"
                        )
                    await asyncio.sleep(check_interval)
                    continue
                result = cv2.matchTemplate(roi, self.retry_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if self.debug:
                    print(f"[结算检测] 匹配值={max_val:.3f}, 阈值={threshold}")
                if max_val >= threshold:
                    print(f"[结算检测] 检测到行动结束 (置信度: {max_val:.3f})")
                    return True
            except Exception as e:
                if self.debug:
                    print(f"[结算检测] 检测异常: {e}")
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
    ):
        if self._abort:
            print("[紧急暂停] 初始化阶段已收到暂停指令，直接退出")
            return
        self._abort = False
        self._stopping = False
        with open(script_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        script = ScriptModel(**data)
        self.executor.load_script(script, borrow_support=borrow_support, direct_start=direct_start)

        print(f"脚本加载完成: {script.stage_name or '未命名'}")
        print(f"地图格子: {script.grid_rows}x{script.grid_cols}")
        print(f"操作数: {len(script.actions)}")

        # 自动选择关卡
        if auto_select_stage and script.stage_name:
            print(f"[自动选关] 尝试进入关卡: {script.stage_name}")
            ok = await self.selector.enter_stage(
                script.stage_name,
                borrow_support=borrow_support,
                support_friend_index=support_friend_index,
                support_skill=support_skill,
                support_module=support_module,
                direct_start=direct_start,
                challenge_mode=challenge_mode,
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

            # 如果检测到漏怪，执行重试流程
            if self._leak_detected:
                # 非无限凸图模式下只补打一次，再次漏怪则停止
                if not loop_mode and self._leak_retried:
                    print("[漏怪检测] 补打后再次漏怪，停止运行")
                    break
                label = "[无限凸图]" if loop_mode else "[漏怪检测]"
                print(f"{label} 检测到漏怪，执行重试...")
                ok = await self.retry_handler.handle_leak_once(script.stage_name, should_stop=lambda: self._stopping)
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
                    script.stage_name,
                    borrow_support=borrow_support,
                    support_friend_index=support_friend_index,
                    support_skill=support_skill,
                    support_module=support_module,
                    direct_start=False,
                    challenge_mode=challenge_mode,
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
        print("用法: python main.py <script.json> [--loop] [--leak] [--debug] [--borrow-support [--support-friend-index N] [--support-skill N] [--support-module N]] [--direct-start] [--challenge-mode] [--cost-tag {normal|cc_25|cc_50|cc_75}] [--pause-key KEY] [--skill-key KEY] [--retreat-key KEY]")
        sys.exit(1)

    loop_mode = "--loop" in sys.argv
    leak_mode = "--leak" in sys.argv
    debug_mode = "--debug" in sys.argv
    borrow_support = "--borrow-support" in sys.argv
    direct_start = "--direct-start" in sys.argv
    challenge_mode = "--challenge-mode" in sys.argv
    if challenge_mode and direct_start:
        print("错误：--challenge-mode（突袭模式）与 --direct-start（直接开始作战）不能同时开启")
        sys.exit(1)

    if loop_mode and direct_start:
        print("错误：--loop（无限凸图）与 --direct-start（直接开始作战）不能同时开启")
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

    if cost_tag and cost_tag not in list_calibrations():
        print(f"错误：--cost-tag 必须是 {list_calibrations()} 之一")
        sys.exit(1)

    pause_key = _arg_str("--pause-key", "p")
    skill_key = _arg_str("--skill-key", "e")
    retreat_key = _arg_str("--retreat-key", "q")
    action.configure_keys(pause=pause_key, skill=skill_key, retreat=retreat_key)

    runner = Runner(debug=debug_mode, cost_tag=cost_tag)
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
        )
    finally:
        # 显式清理本进程注册的全局键盘钩子，避免退出后残留影响 GUI 进程
        try:
            keyboard.unhook_all()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
