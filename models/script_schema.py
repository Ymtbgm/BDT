from enum import Enum
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class ActionType(str, Enum):
    DEPLOY = "deploy"
    RETREAT = "retreat"
    SKILL = "skill"
    SPEED_UP = "speed_up"
    SPEED_DOWN = "speed_down"
    PAUSE = "pause"
    ADD_ITEM = "add_item"  # 在部署区新增额外道具（如击杀奖励装置）


class OperatorAction(BaseModel):
    time_ms: int = Field(..., description="脚本执行时间（毫秒），相对于关卡开始")
    action: ActionType = Field(..., description="操作类型")
    operator_name: Optional[str] = Field(None, description="干员名称")
    grid: Optional[Tuple[int, int]] = Field(None, description="目标格子 (row, col)")
    direction: Optional[str] = Field(None, description="部署方向: up/down/left/right")
    is_object: bool = Field(False, description="是否为场上道具/衍生物，True 时直接对格子操作，不走部署栏流程")


class ItemInfo(BaseModel):
    name: str = Field(..., description="道具名称")
    charges: int = Field(..., description="可使用次数")


class ScriptModel(BaseModel):
    version: str = "1.0"
    stage_code: Optional[str] = Field(None, description="关卡代号，如 1-7，用于精确查询相机位置")
    stage_name: Optional[str] = Field(None, description="关卡名，OCR校验用")
    grid_rows: int = Field(..., description="地图总行数")
    grid_cols: int = Field(..., description="地图总列数")
    operators: List[str] = Field(default_factory=list, description="初始携带干员列表，按位置顺序")
    items: List[ItemInfo] = Field(default_factory=list, description="关卡特殊部署物（道具），优先排列在部署栏最右侧")
    actions: List[OperatorAction] = Field(default_factory=list, description="时间轴操作序列")

    def sort_actions(self):
        self.actions.sort(key=lambda a: a.time_ms)
