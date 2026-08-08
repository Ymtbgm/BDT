"""费用条帧同步校准数据。

危机合约等费用回复 tag 会改变费用条每次回复的“帧数”，导致白像素分布不再
是简单的线性增长。这里保存每个模式下每一帧对应的期望白像素数量，
CostBarSync 通过最近邻匹配来估算当前帧号。

数据通过 tools/capture_cost_bar_cc.py 在子弹时间下截取得到。
"""

from typing import Dict, List, Optional


class CostBarCalibration:
    def __init__(
        self,
        name: str,
        cycle_length: int,
        frame_duration_ms: float,
        expected_counts: List[int],
        zero_ambiguous_until_ms: Optional[float] = None,
    ):
        self.name = name
        self.cycle_length = cycle_length
        self.frame_duration_ms = frame_duration_ms
        self.expected_counts = list(expected_counts)
        self.zero_ambiguous_until_ms = zero_ambiguous_until_ms
        if len(self.expected_counts) != self.cycle_length:
            raise ValueError(
                f"校准表 {name}: expected_counts 长度 {len(self.expected_counts)} "
                f"与 cycle_length {self.cycle_length} 不符"
            )

    def cycle_duration_ms(self) -> float:
        return self.cycle_length * self.frame_duration_ms


# 正常模式：30 帧/秒，费用条 1 秒循环一次。
# 以下数据为实际截取，覆盖 frame 0~29 的白像素分布。
_NORMAL_EXPECTED = [
    0, 48, 90, 138, 180, 222, 270, 312, 360, 402,
    450, 492, 540, 582, 630, 672, 720, 762, 810, 852,
    900, 942, 990, 1032, 1080, 1122, 1170, 1212, 1260, 1302,
]

# 前 10 秒的正常模式：第 29 帧白像素为 0，直接跳到下一秒的第 0 帧。
_NORMAL_EARLY_EXPECTED = _NORMAL_EXPECTED[:-1] + [0]

# 危机合约 tag：游戏实际仍为 30fps，但费用条完成一次回费循环需要更多游戏帧。
# 以下数据为子弹时间下截取，已根据实际白像素分布录入。
_CC_25_EXPECTED = [
    0, 24, 54, 90, 126, 156, 192, 228, 258, 294,
    324, 360, 396, 426, 462, 492, 528, 564, 594, 630,
    660, 696, 732, 762, 798, 834, 864, 900, 930, 966,
    1002, 1032, 1068, 1098, 1134, 1170, 1200, 1236, 1266, 0,
]

# 危机合约 50% tag：60 帧一循环
_CC_50_EXPECTED = [
    0, 0, 24, 48, 66, 90, 114, 138, 156, 180,
    204, 222, 246, 270, 294, 312, 336, 360, 384, 402,
    426, 450, 474, 492, 516, 540, 564, 582, 606, 630,
    654, 672, 696, 720, 738, 762, 786, 810, 828, 852,
    876, 900, 918, 942, 966, 990, 1008, 1032, 1056, 1080,
    1098, 1122, 1146, 1170, 1188, 1212, 1236, 1260, 1278, 0,
]

# 危机合约 75% tag：120 帧一循环
_CC_75_EXPECTED = [
    0, 0, 0, 0, 12, 24, 36, 48, 54, 66,
    78, 90, 102, 114, 126, 138, 144, 156, 168, 180,
    192, 204, 216, 222, 234, 246, 258, 270, 282, 294,
    306, 312, 324, 336, 348, 360, 372, 384, 396, 402,
    414, 426, 438, 450, 462, 474, 480, 492, 504, 516,
    528, 540, 552, 564, 570, 582, 594, 606, 618, 630,
    642, 654, 660, 672, 684, 696, 708, 720, 732, 738,
    750, 762, 774, 786, 798, 810, 822, 828, 840, 852,
    864, 876, 888, 900, 912, 918, 930, 942, 954, 966,
    978, 990, 1002, 1008, 1020, 1032, 1044, 1056, 1068, 1080,
    1086, 1098, 1110, 1122, 1134, 1146, 1158, 1170, 1176, 1188,
    1200, 1212, 1224, 1236, 1248, 1260, 1266, 1278, 1290, 0,
]


COST_BAR_CALIBRATIONS: Dict[str, CostBarCalibration] = {
    "normal": CostBarCalibration(
        name="normal",
        cycle_length=30,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_NORMAL_EXPECTED,
    ),
    "normal_early": CostBarCalibration(
        name="normal_early",
        cycle_length=30,
        frame_duration_ms=1000.0 / 30.0,
        expected_counts=_NORMAL_EARLY_EXPECTED,
        zero_ambiguous_until_ms=10000.0,
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
