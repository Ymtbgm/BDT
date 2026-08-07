from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from models.script_schema import ActionType


class KeyframeType(str, Enum):
    SQUAD_AVATAR = "squad_avatar"
    SQUAD_NAME = "squad_name"
    TEAM_BAR = "team_bar"
    DEPLOY_SLOT = "deploy_slot"
    DEPLOY_BAR = "deploy_bar"
    DEPLOY_NAME_CARD = "deploy_name_card"
    FIELD_SELECTION = "field_selection"


class Keyframe(BaseModel):
    """一次关键帧截图的元数据。"""

    id: str = Field(..., description="关键帧唯一 ID")
    path: str = Field(..., description="PNG 文件相对 raw_recording.json 的路径")
    type: KeyframeType = Field(..., description="关键帧类型")
    time_ms: int = Field(..., description="录制时间（毫秒），相对于关卡开始")
    bar_index: Optional[int] = Field(None, description="部署栏 slot 索引（从右往左 0 开始）")
    grid: Optional[Tuple[int, int]] = Field(None, description="目标格子 (row, col)")
    roi: Optional[Tuple[float, float, float, float]] = Field(
        None, description="关键帧 ROI 相对于窗口的比例 (x_ratio, y_ratio, w_ratio, h_ratio)"
    )


class RawAction(BaseModel):
    """录制阶段未解析的原始操作。"""

    time_ms: int = Field(..., description="录制时间（毫秒），相对于关卡开始")
    action: ActionType = Field(..., description="操作类型")
    target_ref: str = Field(..., description="占位目标，如 __click_0.576923__、__grid_2_3__")
    grid: Optional[Tuple[int, int]] = Field(None, description="目标格子 (row, col)")
    direction: Optional[str] = Field(None, description="部署方向: up/down/left/right")
    keyframe_ids: List[str] = Field(default_factory=list, description="关联的关键帧 ID 列表")


class RawRecording(BaseModel):
    """关键帧录制器的输出，离线解析后生成 ScriptModel。"""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0"
    stage_code: str = Field(..., description="关卡代号，用于查询相机位置和地图尺寸")
    grid_rows: int = Field(..., description="地图总行数，由 stage_code 从 levels.json 自动解析")
    grid_cols: int = Field(..., description="地图总列数，由 stage_code 从 levels.json 自动解析")
    session_id: str = Field(..., description="录制会话 ID，用于关键帧目录命名")
    initial_operator_count: int = Field(0, description="用户输入的初始干员数量")
    initial_item_count: int = Field(0, description="用户输入的初始道具数量")
    keyframes: Dict[str, Keyframe] = Field(default_factory=dict, description="关键帧 ID -> 元数据")
    actions: List[RawAction] = Field(default_factory=list, description="原始操作序列")
    hints: Dict[str, Any] = Field(default_factory=dict, description="可选提示：用户填的 aliases、bindings 等")

    @model_validator(mode="before")
    @classmethod
    def _migrate_and_populate(cls, data: Any) -> Any:
        """旧录制兼容：stage_name 自动迁移为 stage_code；行列从 levels.json 自动补全。"""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        if not data.get("stage_code") and data.get("stage_name"):
            data["stage_code"] = data["stage_name"]

        code = data.get("stage_code")
        if code:
            from core.tile_pos import load_stage_dimensions
            dims = load_stage_dimensions(code)
            if dims:
                width, height = dims
                data["grid_cols"] = width
                data["grid_rows"] = height
            else:
                data.setdefault("grid_cols", 9)
                data.setdefault("grid_rows", 7)

        return data

    def sort_actions(self):
        self.actions.sort(key=lambda a: a.time_ms)
