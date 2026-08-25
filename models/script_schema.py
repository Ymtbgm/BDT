from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ActionType(str, Enum):
    DEPLOY = "deploy"
    RETREAT = "retreat"
    SKILL = "skill"
    SPEED_UP = "speed_up"
    SPEED_DOWN = "speed_down"
    PAUSE = "pause"
    ADD_ITEM = "add_item"  # 在部署区新增额外道具（如击杀奖励装置）
    ADD_SUMMON = "add_summon"  # 在部署区新增特殊召唤物（按费用插入干员区域）
    REMOVE_SUMMON = "remove_summon"  # 从部署区移除特殊召唤物（如干员撤退带走）
    RESET_SUMMON = "reset_summon"  # 强制修正部署栏中特殊召唤物数量（非用户操作，生命周期事件）
    SPECIAL_BEHAVIOR = "special_behavior"  # 特殊行为（如概率点检查）


class OperatorAction(BaseModel):
    time_ms: int = Field(..., description="脚本执行时间（毫秒），相对于关卡开始")
    action: ActionType = Field(..., description="操作类型")
    operator_name: Optional[str] = Field(None, description="干员名称")
    grid: Optional[Tuple[int, int]] = Field(None, description="目标格子 (row, col)")
    direction: Optional[str] = Field(None, description="部署方向: up/down/left/right")
    is_object: bool = Field(False, description="是否为场上道具/衍生物，True 时直接对格子操作，不走部署栏流程")
    params: Optional[Dict[str, Any]] = Field(None, description="特殊行为等所需的额外参数")


class SummonInfo(BaseModel):
    name: str = Field(..., description="召唤物名称")
    cost: int = Field(..., description="部署费用，用于在部署栏中按费用排序定位")
    initial_charges: int = Field(
        default=0,
        description="初始进入部署栏的可用次数；0 表示不在初始栏位，>0 表示初始即存在（无限道具用 1）",
    )


class SummonBinding(BaseModel):
    operator_name: str = Field(..., description="绑定干员名称")
    summon_name: str = Field(..., description="绑定召唤物名称")
    initial_count: int = Field(default=1, description="干员部署后初始进入部署栏的召唤物数量")


class ItemInfo(BaseModel):
    name: str = Field(..., description="道具名称")
    charges: int = Field(..., description="可使用次数")


class ScriptModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    version: str = "1.0"
    stage_code: str = Field(..., description="关卡代号，如 1-7，用于精确查询相机位置和地图尺寸")
    grid_rows: int = Field(..., description="地图总行数，由 stage_code 从 levels.json 自动解析")
    grid_cols: int = Field(..., description="地图总列数，由 stage_code 从 levels.json 自动解析")
    operators: List[str] = Field(default_factory=list, description="初始携带干员列表，按位置顺序")
    items: List[ItemInfo] = Field(default_factory=list, description="关卡特殊部署物（道具），优先排列在部署栏最右侧")
    summons: List[SummonInfo] = Field(default_factory=list, description="特殊召唤物（如无人机、召唤物等），按费用插入干员区域")
    summon_bindings: List[SummonBinding] = Field(
        default_factory=list,
        description="干员与召唤物的绑定关系，执行器在干员撤退时据此清理对应召唤物",
    )
    actions: List[OperatorAction] = Field(default_factory=list, description="时间轴操作序列")
    takeover_boundary_index: Optional[int] = Field(
        default=None,
        description="装载脚本执行与用户接管后录制的分界索引；None 表示无接管或纯录制脚本",
    )

    @model_validator(mode="before")
    @classmethod
    def _migrate_and_populate(cls, data: Any) -> Any:
        """旧脚本兼容：stage_name 自动迁移为 stage_code；行列从 levels.json 自动补全。"""
        if not isinstance(data, dict):
            return data
        data = dict(data)

        # 旧字段兼容：没有 stage_code 时拿 stage_name 顶上
        if not data.get("stage_code") and data.get("stage_name"):
            data["stage_code"] = data["stage_name"]

        code = data.get("stage_code")
        if code:
            from core.map.tile_pos import load_stage_dimensions
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

    def validate_deploy_directions(self) -> List[Tuple[int, str, str]]:
        """检查 DEPLOY 动作是否缺少方向参数。

        返回按 time_ms 排序的三元组列表：(time_ms, 类别, 名称)。
        其中类别为 "干员"/"道具"/"召唤物"，供 UI 格式化为“秒/帧”提示。

        规则：
          1. script.operators 中的干员：所有 DEPLOY 必须有方向。
          2. script.items / script.summons 中的同名道具/召唤物：若超过一半的
             DEPLOY 有方向，则该名称所有 DEPLOY 都必须有方向。
        """
        result: List[Tuple[int, str, str]] = []
        operator_names = set(self.operators or [])
        item_names = {it.name for it in (self.items or [])}
        summon_names = {s.name for s in (self.summons or [])}

        def _label(name: str) -> str:
            if name in item_names:
                return "道具"
            if name in summon_names:
                return "召唤物"
            return "干员"

        # 干员：强制要求方向
        for action in self.actions or []:
            if action.action != ActionType.DEPLOY:
                continue
            name = action.operator_name
            if not name or name not in operator_names:
                continue
            if not action.direction:
                result.append((action.time_ms, "干员", name))

        # 道具/召唤物：按名称统计，多数有方向则认为该名称需要方向
        by_name: Dict[str, List[Tuple[int, OperatorAction]]] = {}
        for action in self.actions or []:
            if action.action != ActionType.DEPLOY:
                continue
            name = action.operator_name
            if not name or name in operator_names:
                continue
            if name not in item_names and name not in summon_names:
                continue
            by_name.setdefault(name, []).append((action.time_ms, action))

        for name, entries in by_name.items():
            total = len(entries)
            directed = sum(1 for _, act in entries if act.direction)
            if directed > total / 2:
                for time_ms, act in entries:
                    if not act.direction:
                        result.append((time_ms, _label(name), name))

        result.sort(key=lambda x: x[0])
        return result
