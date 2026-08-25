import time
from typing import Any, Dict, List, Optional

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper
from core.special_behaviors.config_field import ConfigField

from .base import ProbabilityCheckpointMethod


class GridTargetMethod(ProbabilityCheckpointMethod):
    """格子目标检查：检查指定格子上的单位名称是否与目标一致。"""

    method_id = "grid_target"
    description = "我方单位存在检测"

    # 名称卡 ROI（基于 2560x1600 的绝对像素 x,y,w,h），显示在画面左上角
    _NAME_CARD_X = 0
    _NAME_CARD_Y = 480
    _NAME_CARD_W = 240
    _NAME_CARD_H = 50

    def get_config_fields(self) -> List[ConfigField]:
        return [
            ConfigField(
                name="grid",
                label="目标格子 (行,列)",
                type="str",
                default="",
                hint="例如 3,4",
            ),
            ConfigField(
                name="target_name",
                label="目标单位",
                type="unit",
                default="",
                hint="从干员/道具/召唤物列表中选择要验证的单位名称",
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
        grid_str = params.get("grid", "").strip()
        target_name = (params.get("target_name", "") or "").strip()

        if not grid_str or not target_name:
            print("[格子目标检查] 参数不完整，跳过检查并视为通过")
            return True

        try:
            row, col = map(int, grid_str.split(","))
        except Exception:
            print(f"[格子目标检查] 格子格式错误: {grid_str}")
            return True

        action = context.get("action") if context else None
        ocr = context.get("ocr") if context else None
        pool = context.get("pool") if context else None
        if action is None or ocr is None:
            print("[格子目标检查] 缺少 action/ocr 上下文，无法检测")
            return True

        # 计算目标格子的绝对点击坐标（side 视角下为视觉 tile 中心）
        x, y = grid_mapper.grid_to_pixel(row, col, side=True)
        left = capture.monitor.get("left", 0)
        top = capture.monitor.get("top", 0)
        click_x, click_y = int(x + left), int(y + top)

        print(f"[格子目标检查] 检查格子 ({row},{col})，目标单位: {target_name}")

        # 选中对应格子的单位，进入子弹时间
        print(f"[格子目标检查] 选中目标格子 ({row},{col}) 进入子弹时间: click=({click_x},{click_y})")
        action.select_operator_matchstick(click_x, click_y)
        time.sleep(0.8)

        # 截取名称卡 ROI 并 OCR
        w, h = capture.get_window_size()
        card_x = left + int(self._NAME_CARD_X / 2560 * w)
        card_y = top + int(self._NAME_CARD_Y / 1600 * h)
        card_w = int(self._NAME_CARD_W / 2560 * w)
        card_h = int(self._NAME_CARD_H / 1600 * h)

        try:
            roi = capture.capture_roi(card_x, card_y, card_w, card_h)
        except Exception as e:
            print(f"[格子目标检查] 截图失败: {e}")
            return True

        found = ocr.find_text(roi, target_name)
        passed = found is not None
        print(
            f"[格子目标检查] 名称卡 OCR 结果: {'匹配' if passed else '不匹配'} "
            f"(target={target_name})"
        )

        no_cleanup = context.get("no_cleanup", False) if context else False
        if no_cleanup:
            # 下一个动作是同格子的 SKILL/RETREAT，复用当前选中状态，不退出子弹时间
            print("[格子目标检查] 下一动作同格子，保留选中状态")
            if context is not None:
                context["preselected_grid"] = (row, col)
        else:
            # 检测完成后切回部署栏 0 号位，退出子弹时间，避免影响后续操作
            if pool and action:
                pos0 = pool.get_bar_index_pos(0)
                if pos0:
                    print(f"[格子目标检查] 切回部署栏 0 号位退出子弹时间: pos=({pos0[0]},{pos0[1]})")
                    action.select_at(pos0[0], pos0[1])
                    time.sleep(0.3)

        return passed
