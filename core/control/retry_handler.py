import asyncio
import time
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
import pydirectinput
from core.capture.capture import WindowCapture
from core.control.stage_selector import StageSelector


class StageRetryHandler:
    """漏怪/失败后的自动重试处理器。

    流程：
    1. ROI 区域模板匹配检测漏怪提示或失败提示
    2. 漏怪：点击返回/关闭提示 -> 点击重新挑战（两次）-> 等待加载后再点击
       失败：直接点击重新挑战（两次确认）-> 等待加载后再点击
    3. 调用 StageSelector 重新进入关卡
    """

    # 比例坐标（基于 2560x1600）
    # 漏怪 ROI 区域: x1=1622, y1=26, x2=1715, y2=51
    _LEAK_ROI_RATIO = (1622 / 2560, 26 / 1600, (1715 - 1622) / 2560, (51 - 26) / 1600)
    # 失败提示 ROI 区域: x=228, y=839, w=521, h=70
    _FAILED_ROI_RATIO = (228 / 2560, 839 / 1600, 521 / 2560, 70 / 1600)
    _CLICK_RETURN_RATIO = (131 / 2560, 73 / 1600)
    _CLICK_RETRY_RATIO = (1912 / 2560, 1194 / 1600)

    def __init__(
        self,
        capture: WindowCapture,
        selector: StageSelector,
        template_path: Optional[str] = None,
        failed_template_path: Optional[str] = None,
        mission_end_template_path: Optional[str] = None,
        threshold: float = 0.9,
        mission_end_threshold: float = 0.9,
        debug: bool = False,
    ):
        self.capture = capture
        self.selector = selector
        self.threshold = threshold
        self.mission_end_threshold = mission_end_threshold
        self.debug = debug
        self._template: Optional[np.ndarray] = None
        self._failed_template: Optional[np.ndarray] = None
        self._mission_end_template: Optional[np.ndarray] = None
        if template_path:
            self.load_template(template_path)
        if failed_template_path:
            self.load_failed_template(failed_template_path)
        if mission_end_template_path:
            self.load_mission_end_template(mission_end_template_path)

    @staticmethod
    def _load_image(path: str) -> Optional[np.ndarray]:
        """加载图片，兼容中文路径；失败时返回 None 而非抛异常。"""
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        except Exception:
            img = None
        if img is None:
            print(f"[警告] 模板图片不存在或无法读取: {path}")
            return None
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        return img

    def load_template(self, path: str):
        """加载漏怪提示模板。"""
        self._template = self._load_image(path)

    def load_failed_template(self, path: str):
        """加载失败提示模板。"""
        self._failed_template = self._load_image(path)

    def _ensure_templates_loaded(self):
        """调试用：检查模板是否加载成功。"""
        if self._template is None:
            print("[漏怪检测] 漏怪模板未加载，检测将始终返回 False")
        if self._failed_template is None:
            print("[失败检测] 失败模板未加载，检测将始终返回 False")

    def _ratio_to_pixel(self, rx: float, ry: float) -> Tuple[int, int]:
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return left + int(w * rx), top + int(h * ry)

    def _get_roi(self, ratio: Tuple[float, float, float, float]) -> Tuple[int, int, int, int]:
        """获取指定比例 ROI 的像素坐标 (x, y, w, h)，做边界保护。"""
        w, h = self.capture.get_window_size()
        rx, ry, rw, rh = ratio
        x = int(w * rx)
        y = int(h * ry)
        roi_w = int(w * rw)
        roi_h = int(h * rh)
        # 边界保护
        x = max(0, min(x, w - 1))
        y = max(0, min(y, h - 1))
        roi_w = min(roi_w, w - x)
        roi_h = min(roi_h, h - y)
        return x, y, roi_w, roi_h

    @staticmethod
    def _split_bgr_and_alpha_mask(img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """把 BGRA 拆分为 BGR 图像 + 单通道 mask（alpha>0 为有效区域）。

        若输入为灰度则转成 BGR，mask 全 1；若已是 BGR 则 mask 全 1。
        """
        if img.ndim == 2:
            bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
            return bgr, mask
        if img.ndim == 3 and img.shape[2] == 4:
            bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            mask = (img[:, :, 3] > 0).astype(np.uint8) * 255
            return bgr, mask
        # 3 通道 BGR
        mask = np.ones(img.shape[:2], dtype=np.uint8) * 255
        return img, mask

    def _check_template_in_roi(
        self,
        template: Optional[np.ndarray],
        ratio: Tuple[float, float, float, float],
        label: str,
        threshold: Optional[float] = None,
    ) -> bool:
        """在指定 ROI 区域对模板做带 mask 的匹配，透明部分不计入比较。"""
        if template is None:
            return False
        if threshold is None:
            threshold = self.threshold
        frame = self.capture.capture()
        x, y, roi_w, roi_h = self._get_roi(ratio)
        if roi_w <= 0 or roi_h <= 0:
            return False
        roi = frame[y : y + roi_h, x : x + roi_w]
        if roi.shape[0] < template.shape[0] or roi.shape[1] < template.shape[1]:
            if self.debug:
                print(f"[{label}] ROI({roi_w}x{roi_h}) 小于模板({template.shape[1]}x{template.shape[0]})")
            return False

        # 统一转 BGR，并按 alpha 生成 mask；mask 区域才参与匹配
        roi_bgr, _ = self._split_bgr_and_alpha_mask(roi)
        templ_bgr, templ_mask = self._split_bgr_and_alpha_mask(template)
        result = cv2.matchTemplate(roi_bgr, templ_bgr, cv2.TM_CCORR_NORMED, mask=templ_mask)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        # 处理 ROI 全黑/全白或截图异常导致的 -inf / nan
        if not (isinstance(max_val, (int, float)) and np.isfinite(max_val)):
            if self.debug:
                print(f"[{label}] 匹配值异常: {max_val}, 跳过本次检测")
            return False
        if self.debug:
            print(f"[{label}] 匹配值={max_val:.3f}, 阈值={threshold}, ROI=({x},{y},{roi_w},{roi_h})")
        return max_val > threshold

    def check_leak(self) -> bool:
        """在 ROI 区域执行模板匹配，返回是否检测到漏怪。"""
        return self._check_template_in_roi(self._template, self._LEAK_ROI_RATIO, "漏怪检测")

    # 行动结束/结算界面 ROI（基于 2560x1600）
    _MISSION_END_ROI_RATIO = (102 / 2560, 468 / 1600, 534 / 2560, 192 / 1600)

    def load_mission_end_template(self, path: str):
        """加载行动结束/结算界面模板，用于确认是否已离开结算界面。"""
        self._mission_end_template = self._load_image(path)

    def check_mission_end(self) -> bool:
        """在 ROI 区域执行模板匹配，返回是否检测到行动结束/结算界面。"""
        return self._check_template_in_roi(
            self._mission_end_template,
            self._MISSION_END_ROI_RATIO,
            "结算检测",
            threshold=self.mission_end_threshold,
        )

    def check_failed(self) -> bool:
        """在 ROI 区域执行模板匹配，返回是否检测到失败提示。"""
        return self._check_template_in_roi(self._failed_template, self._FAILED_ROI_RATIO, "失败检测")

    async def _click_return(self):
        """点击返回/关闭提示。"""
        x, y = self._ratio_to_pixel(*self._CLICK_RETURN_RATIO)
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(2.0)

    async def _click_retry(self, after_sleep: float = 2.0):
        """点击重新挑战按钮。"""
        x, y = self._ratio_to_pixel(*self._CLICK_RETRY_RATIO)
        print(f"[重试点击] 点击重新挑战坐标: ({x}, {y}), 等待 {after_sleep}s")
        pydirectinput.moveTo(x, y)
        pydirectinput.mouseDown(button="left")
        await asyncio.sleep(0.05)
        pydirectinput.mouseUp(button="left")
        await asyncio.sleep(after_sleep)

    async def _wait_for_retry_screen_to_clear(
        self, timeout: float = 20.0, interval: float = 0.5
    ) -> bool:
        """等待结算界面消失，确认已返回关卡准备界面再开始 OCR。

        失败界面本身也是结算界面的一种（带"行动结束"），因此只需检测
        mission_end_template（retry.png）即可，避免 failed.png 阈值擦边导致误判。
        若模板未加载或超时未消失，则打印警告并返回 False，由调用方决定是否继续。
        """
        if self._mission_end_template is None:
            print("[重试等待] 结算模板未加载，跳过界面确认")
            return True

        end = time.time() + timeout
        while time.time() < end:
            on_mission_end = self._check_template_in_roi(
                self._mission_end_template, self._MISSION_END_ROI_RATIO, "重试-结算检测"
            )
            if self.debug:
                print(f"[重试等待] 结算界面={on_mission_end}")
            if not on_mission_end:
                print("[重试等待] 已离开结算界面")
                return True
            await asyncio.sleep(interval)

        print(f"[重试等待] 等待 {timeout}s 后仍未离开结算界面，继续执行")
        return False

    async def _perform_failed_retry_clicks(self, sand_table: bool = False):
        """执行失败提示后的重试点击流程（跳过返回提示）。

        沙盘推演模式下只需单次点击即可回到初始界面。
        """
        if sand_table:
            await self._click_retry(after_sleep=5.0)
        else:
            # 失败界面直接显示重新挑战，执行后两次确认点击
            await self._click_retry(after_sleep=8.0)
            await self._click_retry(after_sleep=2.0)

    async def _perform_retry_clicks(self, sand_table: bool = False):
        """执行漏怪后的完整重试点击流程。

        沙盘推演模式下只需单次点击即可回到初始界面。
        """
        # 1. 点击返回/关闭提示
        await self._click_return()
        if sand_table:
            # 2. 单次点击重新挑战
            await self._click_retry(after_sleep=5.0)
        else:
            # 2. 点击重新挑战
            await self._click_retry(after_sleep=2.0)
            # 3. 再点击一次（确认）
            await self._click_retry(after_sleep=8.0)
            # 4. 等待加载后再点击一次
            await self._click_retry(after_sleep=2.0)

    async def run_retry_loop(
        self,
        stage_code: str,
        max_retries: int = 3,
        check_interval: float = 1.0,
    ) -> bool:
        """运行重试循环：检测漏怪 -> 重试点击 -> 重新选关。

        返回 True 表示在某次尝试中成功进入关卡（未检测到漏怪），
        False 表示重试用尽。
        """
        for attempt in range(1, max_retries + 1):
            print(f"[重试] 第 {attempt}/{max_retries} 次尝试...")

            # 先进入关卡
            ok = await self.selector.enter_stage(stage_code)
            if not ok:
                print("[重试] 进入关卡失败，跳过本次尝试")
                continue

            # 等待一段时间让关卡开始（给漏怪检测留出时间窗口）
            # 这里由外层控制，本方法只负责检测和重试
            # 实际上应该在关卡执行过程中检测漏怪
            # 但这里简化为：进入关卡后持续检测

            # 持续检测漏怪，如果检测到就重试
            # 注意：实际使用时应在 executor.run() 并行运行检测
            # 这里提供一个简化版本
            leak_detected = await self._wait_for_leak_or_timeout(check_interval)
            if not leak_detected:
                print("[重试] 本次尝试未检测到漏怪，成功")
                return True

            print("[重试] 检测到漏怪，执行重试流程...")
            await self._perform_retry_clicks()

        print("[重试] 重试次数已用完")
        return False

    async def _wait_for_leak_or_timeout(
        self, check_interval: float = 1.0, timeout: float = 300.0
    ) -> bool:
        """持续检测漏怪直到检测到或超时。"""
        end = time.time() + timeout
        while time.time() < end:
            if self.check_leak():
                return True
            await asyncio.sleep(check_interval)
        return False

    async def handle_leak_once(
        self,
        stage_code: str,
        should_stop=None,
        challenge_mode: bool = False,
        sand_table: bool = False,
        borrow_support: bool = False,
        support_friend_index: Optional[int] = None,
        support_skill: int = 1,
        support_module: int = 1,
    ) -> bool:
        """单次漏怪处理：执行重试点击并重新进入关卡。"""
        if should_stop is not None and should_stop():
            return False
        print("[漏怪处理] 检测到漏怪，开始重试流程...")
        await self._perform_retry_clicks(sand_table=sand_table)
        await self._wait_for_retry_screen_to_clear()
        if should_stop is not None and should_stop():
            return False
        return await self.selector.enter_stage(
            stage_code,
            should_stop=should_stop,
            challenge_mode=challenge_mode,
            sand_table=sand_table,
            borrow_support=borrow_support,
            support_friend_index=support_friend_index,
            support_skill=support_skill,
            support_module=support_module,
        )

    async def handle_failed_once(
        self,
        stage_code: str,
        should_stop=None,
        challenge_mode: bool = False,
        sand_table: bool = False,
        borrow_support: bool = False,
        support_friend_index: Optional[int] = None,
        support_skill: int = 1,
        support_module: int = 1,
    ) -> bool:
        """单次失败处理：执行失败提示后的重试点击并重新进入关卡。"""
        if should_stop is not None and should_stop():
            return False
        print("[失败处理] 检测到失败提示，开始重试流程...")
        await self._perform_failed_retry_clicks(sand_table=sand_table)
        await self._wait_for_retry_screen_to_clear()
        if should_stop is not None and should_stop():
            return False
        return await self.selector.enter_stage(
            stage_code,
            should_stop=should_stop,
            challenge_mode=challenge_mode,
            sand_table=sand_table,
            borrow_support=borrow_support,
            support_friend_index=support_friend_index,
            support_skill=support_skill,
            support_module=support_module,
        )
