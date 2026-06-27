"""费用条帧同步校准数据。

危机合约等费用回复 tag 会改变费用条每次回复的“帧数”，导致白像素分布不再
是简单的线性增长。这里保存每个模式下每一帧对应的期望白像素数量，
CostBarSync 通过最近邻匹配来估算当前帧号。

数据通过 tools/capture_cost_bar_cc.py 在子弹时间下截取得到。
"""

from typing import Dict, List


class CostBarCalibration:
    def __init__(
        self,
        name: str,
        cycle_length: int,
        frame_duration_ms: float,
        expected_counts: List[int],
    ):
        self.name = name
        self.cycle_length = cycle_length
        self.frame_duration_ms = frame_duration_ms
        self.expected_counts = list(expected_counts)
        if len(self.expected_counts) != self.cycle_length:
            raise ValueError(
                f"校准表 {name}: expected_counts 长度 {len(self.expected_counts)} "
                f"与 cycle_length {self.cycle_length} 不符"
            )

    def cycle_duration_ms(self) -> float:
        return self.cycle_length * self.frame_duration_ms


# 正常模式：30 帧/秒，费用条 1 秒循环一次。
# 期望白像素使用线性近似：[0, 45, 90, ..., 1260, 1302]
_NORMAL_EXPECTED = [int(i * 45) for i in range(29)] + [1302]

# 危机合约 tag：游戏实际仍为 30fps，但费用条完成一次回费循环需要更多游戏帧。
# 以下数据为子弹时间下截取，已根据实际白像素分布录入。
_CC_25_EXPECTED = [
    0, 18, 48, 84, 120, 150, 186, 222, 252, 288,
    318, 354, 390, 420, 456, 486, 522, 558, 588, 624,
    654, 690, 726, 756, 792, 828, 858, 894, 924, 960,
    996, 1026, 1062, 1092, 1128, 1164, 1194, 1230, 1260, 0,
]

# 危机合约 50% tag：60 帧一循环
_CC_50_EXPECTED = [
    0, 0, 18, 42, 60, 84, 108, 132, 150, 174,
    198, 216, 240, 264, 288, 306, 330, 354, 378, 396,
    420, 444, 468, 486, 510, 534, 558, 576, 600, 624,
    648, 666, 690, 714, 732, 756, 780, 804, 822, 846,
    870, 894, 912, 936, 960, 984, 1002, 1026, 1050, 1074,
    1092, 1116, 1140, 1164, 1182, 1206, 1230, 1254, 1272, 0,
]

# 危机合约 75% tag：120 帧一循环
_CC_75_EXPECTED = [
    0, 0, 0, 0, 6, 18, 30, 42, 48, 60,
    72, 84, 96, 108, 120, 132, 138, 150, 162, 174,
    186, 198, 210, 216, 228, 240, 252, 264, 276, 288,
    300, 306, 318, 330, 342, 354, 366, 378, 390, 396,
    408, 420, 432, 444, 456, 468, 474, 486, 498, 510,
    522, 534, 546, 558, 564, 576, 588, 600, 612, 624,
    636, 648, 654, 666, 678, 690, 702, 714, 726, 732,
    744, 756, 768, 780, 792, 804, 816, 822, 834, 846,
    858, 870, 882, 894, 906, 912, 924, 936, 948, 960,
    972, 984, 996, 1002, 1014, 1026, 1038, 1050, 1062, 1074,
    1080, 1092, 1104, 1116, 1128, 1140, 1152, 1164, 1170, 1182,
    1194, 1206, 1218, 1230, 1242, 1254, 1260, 1272, 1284, 0,
]


COST_BAR_CALIBRATIONS: Dict[str, CostBarCalibration] = {
    "normal": CostBarCalibration(
        name="normal",
        cycle_length=30,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_NORMAL_EXPECTED,
    ),
    "cc_25": CostBarCalibration(
        name="cc_25",
        cycle_length=40,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_CC_25_EXPECTED,
    ),
    "cc_50": CostBarCalibration(
        name="cc_50",
        cycle_length=60,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_CC_50_EXPECTED,
    ),
    "cc_75": CostBarCalibration(
        name="cc_75",
        cycle_length=120,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_CC_75_EXPECTED,
    ),
}


def get_calibration(name: str) -> CostBarCalibration:
    if name not in COST_BAR_CALIBRATIONS:
        raise ValueError(f"未知的费用条校准模式: {name}，可用: {list(COST_BAR_CALIBRATIONS.keys())}")
    cal = COST_BAR_CALIBRATIONS[name]
    if all(c == 0 for c in cal.expected_counts):
        raise ValueError(f"费用条校准模式 {name} 尚未录入实际数据")
    return cal


def list_calibrations() -> List[str]:
    return [
        name
        for name, cal in COST_BAR_CALIBRATIONS.items()
        if not all(c == 0 for c in cal.expected_counts)
    ]
