import time
import asyncio
from typing import List, Optional, Tuple
from core.capture import WindowCapture
from core.ocr_engine import OCREngine


class StageSelector:
    """关卡选择器：在关卡列表界面定位并进入指定关卡。"""

    # 相对于 2560x1600 客户区的比例坐标
    # 第1步：点击关卡名 -> 第2步：点击"开始行动" -> 第3步：点击"开始"
    START_ACTION_RATIO: Tuple[float, float] = (1953 / 2560, 1487 / 1600)
    START_CONFIRM_RATIO: Tuple[float, float] = (2117 / 2560, 1040 / 1600)
    CHALLENGE_MODE_RATIO: Tuple[float, float] = (1565 / 2560, 1478 / 1600)

    # 助战干员选择相关比例坐标（基于 2560x1600）
    SUPPORT_BUTTON_RATIO: Tuple[float, float] = (2127 / 2560, 527 / 1600)
    SUPPORT_FRIEND_BASE_RATIO: Tuple[float, float] = (330 / 2560, 1242 / 1600)
    SUPPORT_FRIEND_DELTA_X: float = 256 / 2560
    SUPPORT_SKILL_RATIOS: List[Tuple[float, float]] = [
        (1559 / 2560, 698 / 1600),  # 技能1
        (1764 / 2560, 698 / 1600),  # 技能2
        (1967 / 2560, 698 / 1600),  # 技能3
    ]
    SUPPORT_MODULE_RATIOS: List[Tuple[float, float]] = [
        (1562 / 2560, 1001 / 1600),  # 模组1
        (1778 / 2560, 1001 / 1600),  # 模组2
        (1980 / 2560, 1001 / 1600),  # 模组3
    ]
    SUPPORT_CONFIRM_RATIO: Tuple[float, float] = (1679 / 2560, 1235 / 1600)

    def __init__(self, capture: WindowCapture, ocr: OCREngine, debug: bool = False, max_side: int = 640):
        self.capture = capture
        self.ocr = ocr
        self.debug = debug
        # 为适配不同 OCR 后端速度，可调节最长边；
        # transformers CPU 建议 640，paddle_static 可轻松跑到 1280/1920
        self.max_side = max_side

    def _ratio_to_pixel(self, rx: float, ry: float) -> Tuple[int, int]:
        """将相对比例坐标转换为实际屏幕像素坐标。"""
        w, h = self.capture.get_window_size()
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return left + int(w * rx), top + int(h * ry)

    def find_stage(self, stage_name: str) -> Optional[Tuple[int, int]]:
        """在当前截图中查找关卡名并返回屏幕坐标。

        为加速 CPU 推理，整图按比例缩小到最长边 max_side，
        OCR 完成后将坐标放大回原始尺寸。
        """
        import cv2
        frame = self.capture.capture()
        h, w = frame.shape[:2]
        max_side = max(h, w)
        scale = 1.0
        if max_side > self.max_side:
            scale = self.max_side / max_side
            frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )

        all_lines = self.ocr.recognize(frame)
        result = self.ocr.find_text(frame, stage_name, lines=all_lines)
        if result is None:
            return None
        cx, cy, matched_text, conf = result
        # 缩放回原始坐标
        cx = int(cx / scale)
        cy = int(cy / scale)
        left = self.capture.monitor.get("left", 0)
        top = self.capture.monitor.get("top", 0)
        return cx + left, cy + top

    async def click_stage(self, stage_name: str, timeout: float = 60.0, interval: float = 1.0, should_stop=None) -> bool:
        """循环查找关卡名并点击，直到成功或超时。

        注意：为避免 OCR 推理耗时过长导致"刚找到就超时"，
        只有在 loop 开头检查超时；一旦 find_stage 返回有效坐标，
        无论是否已超时都会执行点击。
        """
        import pydirectinput
        end_time = time.time() + timeout
        attempt = 0
        while True:
            if should_stop is not None and should_stop():
                return False
            if time.time() >= end_time:
                break
            attempt += 1
            pos = self.find_stage(stage_name)
            if pos is not None:
                x, y = pos
                pydirectinput.moveTo(x, y)
                pydirectinput.click(button='left')
                return True
            remaining = max(0, int(end_time - time.time()))
            if remaining <= 0:
                break
            await asyncio.sleep(interval)
        if self.debug:
            print(f"[DEBUG] 未找到关卡 '{stage_name}'，截图与 OCR 结果已保存到 debug/ 目录")
        return False

    async def enter_stage(
        self,
        stage_name: str,
        borrow_support: bool = False,
        support_friend_index: Optional[int] = None,
        support_skill: int = 1,
        support_module: int = 1,
        direct_start: bool = False,
        challenge_mode: bool = False,
        should_stop=None,
    ) -> bool:
        """完整流程：查找关卡 -> 点击 -> [可选突袭切换] -> [可选助战选择] -> 点击确认开始。

        若传入 direct_start=True，则跳过 OCR 查找关卡和"开始行动"点击，
        直接等待 2 秒后点击"确认开始"，适用于用户已手动进入准备界面的场景。

        should_stop: 可选的无参可调用对象，返回 True 时立即终止流程并返回 False。
        """
        import pydirectinput

        def _check_stop() -> bool:
            return should_stop is not None and should_stop()

        if direct_start:
            print("[关卡选择] 直接开始作战模式，跳过 OCR 与开始行动点击")
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            x, y = self._ratio_to_pixel(*self.START_CONFIRM_RATIO)
            pydirectinput.moveTo(x, y)
            pydirectinput.click(button='left')
            print("[关卡选择] 已点击确认开始")
            return True

        # 1. 查找并点击关卡名
        found = await self.click_stage(stage_name, should_stop=should_stop)
        if not found:
            print(f"[关卡选择] 未找到关卡: {stage_name}")
            return False
        print(f"[关卡选择] 已点击关卡: {stage_name}")

        # 1.5 突袭模式：点击切换按钮
        if challenge_mode:
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            x, y = self._ratio_to_pixel(*self.CHALLENGE_MODE_RATIO)
            pydirectinput.moveTo(x, y)
            pydirectinput.click(button='left')
            print("[关卡选择] 突袭模式，已点击切换按钮")

        # 2. 等待并点击"开始行动"
        await asyncio.sleep(2.0)
        if _check_stop():
            return False
        x, y = self._ratio_to_pixel(*self.START_ACTION_RATIO)
        pydirectinput.moveTo(x, y)
        pydirectinput.click(button='left')
        print("[关卡选择] 已点击开始行动")

        if borrow_support:
            # 点击"借用干员"按钮
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            x, y = self._ratio_to_pixel(*self.SUPPORT_BUTTON_RATIO)
            pydirectinput.moveTo(x, y)
            pydirectinput.click(button='left')
            print("[关卡选择] 已点击借用干员按钮")

            # 选择好友位置
            await asyncio.sleep(3.0)
            if _check_stop():
                return False
            if support_friend_index is not None and 0 <= support_friend_index <= 8:
                rx = self.SUPPORT_FRIEND_BASE_RATIO[0] + support_friend_index * self.SUPPORT_FRIEND_DELTA_X
                ry = self.SUPPORT_FRIEND_BASE_RATIO[1]
                x, y = self._ratio_to_pixel(rx, ry)
                pydirectinput.moveTo(x, y)
                pydirectinput.click(button='left')
                print(f"[关卡选择] 已选择好友位置 {support_friend_index}")
            else:
                print(f"[关卡选择] 警告: 好友位置 {support_friend_index} 无效，跳过选择")

            # 选择携带技能
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            skill_idx = support_skill - 1
            if 0 <= skill_idx < len(self.SUPPORT_SKILL_RATIOS):
                x, y = self._ratio_to_pixel(*self.SUPPORT_SKILL_RATIOS[skill_idx])
                pydirectinput.moveTo(x, y)
                pydirectinput.click(button='left')
                print(f"[关卡选择] 已选择技能 {support_skill}")
            else:
                print(f"[关卡选择] 警告: 技能 {support_skill} 无效，跳过选择")

            # 选择模组
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            module_idx = support_module - 1
            if 0 <= module_idx < len(self.SUPPORT_MODULE_RATIOS):
                x, y = self._ratio_to_pixel(*self.SUPPORT_MODULE_RATIOS[module_idx])
                pydirectinput.moveTo(x, y)
                pydirectinput.click(button='left')
                print(f"[关卡选择] 已选择模组 {support_module}")
            else:
                print(f"[关卡选择] 警告: 模组 {support_module} 无效，跳过选择")

            # 确认助战选择
            await asyncio.sleep(2.0)
            if _check_stop():
                return False
            x, y = self._ratio_to_pixel(*self.SUPPORT_CONFIRM_RATIO)
            pydirectinput.moveTo(x, y)
            pydirectinput.click(button='left')
            print("[关卡选择] 已确认助战选择")

        # 3. 等待并点击确认开始
        await asyncio.sleep(2.0)
        if _check_stop():
            return False
        x, y = self._ratio_to_pixel(*self.START_CONFIRM_RATIO)
        pydirectinput.moveTo(x, y)
        pydirectinput.click(button='left')
        print("[关卡选择] 已点击确认开始")

        return True
