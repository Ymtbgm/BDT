"""项目通用常量集合。

集中管理 ROI、阈值、键位、时间等硬编码参数，便于协作开发时统一调整。
"""

from typing import Dict, List, Optional, Tuple

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

# “费用不自然回复”模式：区域 B 1x 图标从高亮（正式开始）变为可计时的白像素阈值
# 暗色待机时白像素约 80，高亮后约 1500；取 1000 避免中间态/过渡动画误触发。
REGION_B_BRIGHT_THRESHOLD: int = 1000
# “费用不自然回复”模式：检测到高亮 1x 后的启动时间补偿（ms），需实测校准
NO_REGEN_STARTUP_OFFSET_MS: float = 0.0

# 游戏内倍率
FAST_RATE: float = 1.0
FAST2X_RATE: float = 2.0
SLOW_RATE: float = 0.2

# 帧时间与补偿
FRAME_MS: float = 33.333
STARTUP_OFFSET_MS: float = 50.0

# 区域 B 倍率采样间隔：截图本身约 6ms，间隔应略大于截图耗时；
# 取 1/4 帧（约 8.33ms）可在用户操作后较快响应，同时避免 7ms 时的抖动
RATE_SAMPLER_INTERVAL_MS: float = FRAME_MS / 4.0

# 区域 B 模板匹配
RATE_TEMPLATE_FAST_NAME: str = "1X.png"
RATE_TEMPLATE_FAST2X_NAME: str = "2X.png"
RATE_TEMPLATE_SLOW_NAME: str = "0.2X.png"
RATE_TEMPLATE_MATCH_CONFIDENCE: float = 0.85
RATE_TEMPLATE_TRANSITION_CONFIDENCE: float = 0.70
# 模板掩膜阈值：alpha > 此值或灰度 > 此值视为图标前景
RATE_TEMPLATE_MASK_THRESHOLD: int = 128
# 帧间平均差分超过此值认为 UI 仍在动画/淡化，辅助判定过渡态
RATE_TEMPLATE_DIFF_THRESHOLD: float = 3.0

# TimeKeeper 鼠标事件回查窗口：倍率切换时允许回退到上一个 tick 的最大距离
RATE_CLICK_LOOKBACK_MS: float = 40.0

# 经验补偿：默认设为 0，过渡态按目标倍率累加后通常不需要额外补偿
SLOW_TO_FAST_COMPENSATION_FRAMES: float = 0.0
FAST_TO_SLOW_COMPENSATION_FRAMES: float = 0.0
RATE_TRANSITION_COOLDOWN_FRAMES: int = 5

# 高精度计时器（TimeKeeper）
TIMER_HIGH_PRECISION_INTERVAL_MS: float = 4.0
TIMER_HIGH_PRECISION_SLEEP_RATIO: float = 0.5
TIMER_RESOLUTION_MS: int = 1

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

COST_BAR_THRESHOLD: int = 150
COST_BAR_STEP_PIXELS: float = 45.0
COST_BAR_FULL_PIXELS: int = 1302
COST_BAR_FRAME_OFFSET_MS: float = 0.0

# 帧同步容差：白像素数量与期望值的允许偏差（步长的 70%）
COST_BAR_TOLERANCE_RATIO: float = 0.7

# 费用条同步修正
COST_BAR_SYNC_INTERVAL_MS: float = 100.0
COST_BAR_SYNC_MAX_DIFF_MS: float = 500.0
COST_BAR_SYNC_MAX_SKIP_FRAMES: int = 4  # 费用条帧同步单次最多跳帧数
COST_MAX_TEMPLATE_NAME: str = "cost_max.png"
COST_MAX_MATCH_CONFIDENCE: float = 0.85

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

# 部署栏数量条 OCR 预处理：数量角标更容易混入 UI 浅灰边缘，使用更高阈值
DEPLOY_BAR_QUANTITY_WHITE_THRESHOLD: int = 180
# invert 模式下将 >= 该值的像素转为黑字；当前尝试 250 以减少头像高光误检
DEPLOY_BAR_QUANTITY_INVERT_THRESHOLD: int = 250

# 部署栏头像截取参数（基于 2560x1600）
# 头像基础高度；实际截取尺寸会按当前分辨率等比缩放，并受格子宽度限制
DEPLOY_BAR_AVATAR_SIZE_RATIO: float = 120 / 1600
# 头像中心相对 OperatorPool 计算出的点击纵坐标上移比例（以头像高度为单位）
# 正数向上移、负数向下移；当前按 1600h 下额外下移 10px 标定
DEPLOY_BAR_AVATAR_Y_OFFSET_RATIO: float = -10 / 120

# ============================================================
# 信息录入（编队界面 OCR）
# ============================================================

# 编队界面 13 个干员名的 ROI 比例（x, y, w, h），基于 2560x1600
# 顺序与游戏中实际一致：上 0,2,4,6,8,10 / 下 1,3,5,7,9,11 / 助战 12
SQUAD_NAME_ROI_RATIOS: List[Tuple[float, float, float, float]] = [
    # 上 0 / 下 1
    (326 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (326 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 上 2 / 下 3
    (606 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (606 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 上 4 / 下 5
    (886 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (886 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 上 6 / 下 7
    (1166 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (1166 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 上 8 / 下 9
    (1446 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (1446 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 上 10 / 下 11
    (1726 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
    (1726 / 2560, 1254 / 1600, 230 / 2560, 42 / 1600),
    # 助战 12
    (2039 / 2560, 740 / 1600, 230 / 2560, 42 / 1600),
]

# 编队界面 13 个干员头像的 ROI 比例（x, y, w, h），基于 2560x1600
# 顺序与 SQUAD_NAME_ROI_RATIOS 一致
SQUAD_AVATAR_ROI_RATIOS: List[Tuple[float, float, float, float]] = [
    # 上 0 / 下 1
    (380 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (380 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 上 2 / 下 3
    (660 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (660 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 上 4 / 下 5
    (940 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (940 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 上 6 / 下 7
    (1220 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (1220 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 上 8 / 下 9
    (1500 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (1500 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 上 10 / 下 11
    (1780 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
    (1780 / 2560, 920 / 1600, 120 / 2560, 120 / 1600),
    # 助战 12
    (2060 / 2560, 385 / 1600, 120 / 2560, 120 / 1600),
]

# ============================================================
# 脚本执行 (ScriptExecutor)
# ============================================================

# 技能可点击检测最大跳帧等待次数，每次约前进 1 游戏帧
SKILL_CLICK_MAX_ATTEMPTS: int = 5

# 技能可点击检测置信度阈值
SKILL_CLICK_CONF_THRESH: float = 0.5

# 技能状态图标 ROI 中心与 view_side 的线性关系（基于 2560x1600）。
# 公式：center = coef[0]*vx + coef[1]*vy + coef[2]*vz + coef[3]
# view_side = (vx, vy, vz) 来自 TilePosCalculator.view_side。
SKILL_STATUS_CENTER_COEF_X: Tuple[float, float, float, float] = (
    -151.157764, 5.370037, -13.298396, 1495.365037
)
SKILL_STATUS_CENTER_COEF_Y: Tuple[float, float, float, float] = (
    15.504996, 129.462123, -71.538437, 705.904958
)

# 高台单位技能状态图标相对地面的 Y 偏移（基于 2560x1600 的像素）
SKILL_STATUS_HIGH_Y_OFFSET: int = -25

# 技能状态图标 ROI 尺寸（基于 2560x1600 的像素 w, h）
SKILL_STATUS_ROI_SIZE: Tuple[int, int] = (100, 100)

# 最左三列（第 0-2 列）的 RETREAT/SKILL 操作提前触发时间
LEFT_COLS_ADVANCE_MS: int = 18

# 装载脚本模式下，最左三列 RETREAT/SKILL 的额外提前量。
# 此时计时器直接返回游戏时间，2x 下同样的现实延迟对应双倍游戏时间，
# 因此需要比 standalone 执行器的 18ms 更大。
LOADED_SCRIPT_LEFT_COLS_ADVANCE_MS: int = 27

# _execute_cluster 中推进一帧的计时器补偿
ADVANCE_FRAME_MS: float = 33.0

# wait_until 最后自旋等待阈值
WAIT_SPIN_THRESHOLD_MS: int = 5

# 二倍速下 wait_until 目标提前量（压缩时间 ms），用于抵消暂停键延迟导致的触发偏晚
# 设为 0 表示默认不提前；若实测二倍速仍偏晚，可适量调大
TWOX_EARLY_TRIGGER_MS: int = 0

# 装载脚本模式下（RegionStateTimer 已返回缩放游戏时间）的 wait_until 提前量，
# 用于抵消空格键到游戏真正暂停的延迟。按帧估算：30fps 下 2 帧约 66ms。
LOADED_SCRIPT_EARLY_TRIGGER_MS: int = 66

# ============================================================
# 键位与热键 (action.py 默认配置)
# ============================================================

DEFAULT_PAUSE_KEY: str = "space"
DEFAULT_SKILL_KEY: str = "e"
DEFAULT_RETREAT_KEY: str = "q"
DEFAULT_SPEED_KEY: str = "f"

DEFAULT_MATCHSTICK_HOTKEYS: Dict[str, str] = {
    "select_operator": "r",
    "pass_166ms": "p",
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
