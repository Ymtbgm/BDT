from typing import Any, Dict, List, Optional

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper
from core.special_behaviors.config_field import ConfigField

from .base import ProbabilityCheckpointMethod


class KillCountMethod(ProbabilityCheckpointMethod):
    """击杀数检查：检查当前击杀数是否达到目标值，未达到则重开关卡。

    游戏内击杀数显示格式示例："60/121"（当前击杀/敌方总数）。
    ROI 基于 2560x1600 固定为 (1114, 72, 152, 43)。
    """

    method_id = "kill_count"
    description = "击杀数检测"

    # 击杀数 ROI（基于 2560x1600 的绝对像素 x,y,w,h）
    _KILL_COUNT_X = 1086
    _KILL_COUNT_Y = 26
    _KILL_COUNT_W = 173
    _KILL_COUNT_H = 54

    def get_config_fields(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="target_kill_count",
                label="目标击杀数",
                type="int",
                default=0,
                hint="仅当当前击杀数等于该值时视为通过",
            ),
            ConfigField(
                name="target_total_enemies",
                label="敌方总单位数",
                type="int",
                default=0,
                hint="用于校验 OCR 识别出的总单位数，填 0 表示不校验总数",
            ),
        ]

    def execute(
        self,
        capture: WindowCapture,
        grid_mapper: GridMapper,
        params: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        params = params or {}
        target_kill_count = int(params.get("target_kill_count", 0) or 0)
        target_total_enemies = int(params.get("target_total_enemies", 0) or 0)

        if target_kill_count <= 0:
            print("[击杀数检查] 目标击杀数未设置，跳过检查并视为通过")
            return True

        ocr = context.get("ocr") if context else None
        if ocr is None:
            print("[击杀数检查] 缺少 ocr 上下文，无法检测")
            return True

        left = capture.monitor.get("left", 0)
        top = capture.monitor.get("top", 0)
        w, h = capture.get_window_size()

        roi_x = left + int(self._KILL_COUNT_X / 2560 * w)
        roi_y = top + int(self._KILL_COUNT_Y / 1600 * h)
        roi_w = int(self._KILL_COUNT_W / 2560 * w)
        roi_h = int(self._KILL_COUNT_H / 1600 * h)

        try:
            roi = capture.capture_roi(roi_x, roi_y, roi_w, roi_h)
        except Exception as e:
            print(f"[击杀数检查] 截图失败: {e}")
            return True

        # 先用 OCR 提取整段文本，再解析 "current/total" 格式
        lines = ocr.recognize(roi)
        text = " ".join(line[1][0] for line in lines if line)
        print(f"[击杀数检查] ROI OCR 原始文本: {text!r}")

        current, total = self._parse_kill_count(text)
        print(
            f"[击杀数检查] 解析结果: current={current}, total={total}, "
            f"target={target_kill_count}, target_total={target_total_enemies}"
        )

        if current is None:
            print("[击杀数检查] 未能解析击杀数，视为不通过")
            return False

        if target_total_enemies > 0 and total != target_total_enemies:
            print(
                f"[击杀数检查] 敌方总数校验失败: 识别到 {total}, 期望 {target_total_enemies}"
            )
            return False

        passed = current == target_kill_count
        print(
            f"[击杀数检查] 结果: {'通过' if passed else '不通过'} "
            f"({current}/{total or '?'}, 目标={target_kill_count})"
        )
        return passed

    def _parse_kill_count(self, text: str) -> tuple:
        """从 OCR 文本中解析当前击杀数和总单位数。

        支持格式："60/121"、"60 / 121"、"60/ 121" 等。
        返回 (current, total)，解析失败返回 (None, None)。
        """
        if not text:
            return None, None

        # 去除空白并尝试分割
        cleaned = text.replace(" ", "").replace("\\", "/").strip()
        parts = cleaned.split("/")
        if len(parts) != 2:
            # 尝试用空格分割
            parts = text.split()
            if len(parts) != 2:
                return None, None

        try:
            current = int(parts[0])
            total = int(parts[1])
            return current, total
        except ValueError:
            return None, None
