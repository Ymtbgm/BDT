"""项目通用常量集合。

集中管理 ROI、阈值、键位、时间等硬编码参数，便于协作开发时统一调整。
"""

from typing import Dict, Tuple

# ============================================================
# 区域计时器 (RegionStateTimer)
# ============================================================

# 默认 ROI 基于 2560x1600 的绝对屏幕坐标 (x, y, w, h)
DEFAULT_ROI_A: Tuple[int, int, int, int] = (2375, 53, 112, 88)
DEFAULT_ROI_B: Tuple[int, int, int, int] = (2175, 34, 128, 119)

# 灰度阈值：像素灰度 > threshold 视为白像素
REGION_WHITE_THRESHOLD: int = 200

# 区域 B 白像素阈值（带迟滞）
REGION_B_FAST_THRESHOLD: int = 1200  # > 此值视为 1.0x
REGION_B_SLOW_THRESHOLD: int = 1000  # < 此值视为 0.2x

# 游戏内倍率
FAST_RATE: float = 1.0
SLOW_RATE: float = 0.2

# 帧时间与补偿
FRAME_MS: float = 33.333
STARTUP_OFFSET_MS: float = 50.0
SLOW_TO_FAST_COMPENSATION_FRAMES: float = 1.6
FAST_TO_SLOW_COMPENSATION_FRAMES: float = 0.4
RATE_TRANSITION_COOLDOWN_FRAMES: int = 5

# 键盘事件防抖与保护期
PAUSE_KEY_DEBOUNCE_MS: float = 100.0       # 暂停键 100ms 防抖
MATCHSTICK_SHIELD_MS: float = 400.0        # 划火柴 P/ESC 保护期
MATCHSTICK_HOTKEY_COMPENSATION_MS: float = 0.3  # 划火柴热键触发后的时间补偿

# ============================================================
# 费用条同步 (CostBarSync)
# ============================================================

# 默认 ROI 比例基于 2560x1600 分辨率下费用条位置
COST_BAR_ROI_RATIOS: Tuple[float, float, float, float] = (
    2343 / 2560,           # x
    1278 / 1600,           # y
    (2560 - 2343) / 2560,  # w
    (1284 - 1278) / 1600,  # h
)

COST_BAR_THRESHOLD: int = 200
COST_BAR_STEP_PIXELS: float = 45.0
COST_BAR_FULL_PIXELS: int = 1302
COST_BAR_FRAME_OFFSET_MS: float = 0.0

# 帧同步容差：白像素数量与期望值的允许偏差（步长的 70%）
COST_BAR_TOLERANCE_RATIO: float = 0.7

# ============================================================
# 部署栏 OCR (OperatorPool)
# ============================================================

# 部署栏费用 ROI 比例（基于 2560x1600），覆盖干员头像下方费用数字区域
# y 轴像素范围 1390-1426，仅识别干员费用数字
DEPLOY_BAR_COST_ROI_RATIOS: Tuple[float, float, float, float] = (
    0.0,             # x（从最左侧开始）
    1390 / 1600,     # y
    1.0,             # w（覆盖整个横向区域）
    36 / 1600,       # h
)

# OCR 识别费用数字的最低置信度
DEPLOY_BAR_COST_CONFIDENCE: float = 0.6

# 部署栏费用 OCR 预处理：灰度阈值，低于此值的像素置黑以强化白字
DEPLOY_BAR_COST_WHITE_THRESHOLD: int = 100

# ============================================================
# 脚本执行 (ScriptExecutor)
# ============================================================

# 最左三列（第 0-2 列）的 RETREAT/SKILL 操作提前触发时间
LEFT_COLS_ADVANCE_MS: int = 18

# _execute_cluster 中推进一帧的计时器补偿
ADVANCE_FRAME_MS: float = 33.0

# wait_until 最后自旋等待阈值
WAIT_SPIN_THRESHOLD_MS: int = 5

# ============================================================
# 键位与热键 (action.py 默认配置)
# ============================================================

DEFAULT_PAUSE_KEY: str = "p"
DEFAULT_SKILL_KEY: str = "e"
DEFAULT_RETREAT_KEY: str = "q"

DEFAULT_MATCHSTICK_HOTKEYS: Dict[str, str] = {
    "select_operator": "r",
    "pass_166ms": "space",
    "pass_50ms": "f",
}

# 划火柴 P+ESC 组合的默认停顿时间
MATCHSTICK_INTERVAL_TIME: float = 0.166

# pydirectinput 按键默认按下时长
KEY_PRESS_DURATION: float = 0.05

# ============================================================
# 分辨率基准
# ============================================================

BASE_WIDTH: int = 2560
BASE_HEIGHT: int = 1600

# ============================================================
# stdout 跨进程协议标记
# ============================================================

TIMER_SHIELD_MARKER: str = "__TIMER_SHIELD__"
TIMER_ADJUST_MARKER: str = "__TIMER_ADJUST__"
