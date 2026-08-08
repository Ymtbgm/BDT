import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, model_validator


class SummonClass(str, Enum):
    A = "A"
    B = "B"


class SummonDefinition(BaseModel):
    """召唤物定义。

    通用字段：
      - name: 召唤物名称
      - cost: 部署费用
      - summon_class: A 类（撤退/超时返回）或 B 类（技能/周期补充）
      - initial_count: 绑定干员部署后初始进入部署栏的数量

    Class A 专用：
      - max_return_time_ms: 召唤物上场后多少毫秒自动返回部署栏
      - return_count: 每次返回补充的数量

    Class A 技能锁定变体（字段开关）：
      - skill_lock_enabled: 是否启用技能锁定行为（常态仍为 Class A）
      - skill_auto_trigger_delay_ms: 干员部署后多少毫秒自动触发技能（未设置或 0 表示不自动触发）
      - skill_extra_charges: 绑定干员开启技能时额外补充的召唤物数量
      - skill_end_reset_count: 技能结束时强制将部署栏中该召唤物数量修正为多少
      - skill_duration_ms: 技能持续时间（毫秒），Class B 与 Class A 技能锁定变体共用
      - skill_state_match_enabled: 是否通过头顶 ROI Logo 匹配判断技能结束（弹药/手动关闭类）
      - skill_state_roi_offset_x/y: ROI 相对于干员 normal view 屏幕中心的偏移
      - skill_state_roi_width/height: ROI 尺寸
      - skill_state_active_template: 弹药技能期间出现的固定 logo 模板路径（相对 resource/gui_template）
      - skill_state_inactive_template: 技能准备完毕 logo 模板路径（相对 resource/gui_template），可选
      - skill_state_active_threshold/inactive_threshold: 模板匹配阈值
      - skill_state_consecutive_frames: 连续多少帧确认状态切换
      - skill_end_by_state_match: 为 True 时忽略 skill_duration_ms，用状态匹配触发结束重置

    Class B 专用：
      - gain_interval_ms: 周期补充间隔（毫秒），未设置或 0 表示不周期补充
      - gain_count: 每次补充的数量（兼容旧配置）
      - periodic_gain_count: 每次周期补充的数量，默认回退到 gain_count
      - gain_on_skill: 兼容旧配置，等价于 gain_on_skill_start
      - gain_on_skill_start: 技能开启时是否补充
      - gain_on_skill_end: 技能结束时是否补充
      - skill_duration_ms: 技能持续时间（毫秒），用于计算 gain_on_skill_end 的触发时刻
      - skill_gain_count: 每次技能补充的数量（开启或结束都使用），默认回退到 gain_count
      - max_charges: 部署栏中该召唤物的数量上限

    别名：
      - aliases: 干员名/别名列表，用于信息录入时自动匹配 operator -> summon
      - is_default_alias: 当多个召唤物命中同一别名时，优先选择 is_default_alias=True 的项
    """

    name: str
    cost: int
    summon_class: SummonClass
    initial_count: int = 1

    # Class A
    max_return_time_ms: Optional[int] = None
    return_count: int = 1

    # Class A 技能锁定变体
    skill_lock_enabled: bool = False
    skill_auto_trigger_delay_ms: Optional[int] = None
    skill_extra_charges: int = 0
    skill_end_reset_count: int = 1

    # 头顶 ROI Logo 状态检测（用于弹药/手动关闭类技能）
    skill_state_match_enabled: bool = False
    skill_state_roi_offset_x: int = -35
    skill_state_roi_offset_y: int = -240
    skill_state_roi_width: int = 75
    skill_state_roi_height: int = 75
    skill_state_active_template: Optional[str] = None
    skill_state_inactive_template: Optional[str] = None
    skill_state_active_threshold: float = 0.80
    skill_state_inactive_threshold: float = 0.70
    skill_state_consecutive_frames: int = 5
    skill_end_by_state_match: bool = False

    # Class B
    gain_interval_ms: Optional[int] = None
    gain_count: int = 1
    periodic_gain_count: Optional[int] = None
    gain_on_skill: bool = False
    gain_on_skill_start: bool = False
    gain_on_skill_end: bool = False
    skill_duration_ms: Optional[int] = None
    skill_gain_count: Optional[int] = None
    max_charges: Optional[int] = None

    # 别名匹配
    aliases: List[str] = Field(default_factory=list)
    is_default_alias: bool = False

    @model_validator(mode="after")
    def _apply_gain_count_fallback(self):
        """旧配置兼容：未指定新字段时回退到旧字段。"""
        if self.periodic_gain_count is None:
            self.periodic_gain_count = self.gain_count
        if self.skill_gain_count is None:
            self.skill_gain_count = self.gain_count
        # 旧配置 gain_on_skill 等价于 gain_on_skill_start
        if self.gain_on_skill and not self.gain_on_skill_start and not self.gain_on_skill_end:
            self.gain_on_skill_start = True
        return self

    def is_class_a(self) -> bool:
        return self.summon_class == SummonClass.A

    def is_class_b(self) -> bool:
        return self.summon_class == SummonClass.B

    def is_skill_locked_class_a(self) -> bool:
        """是否为启用技能锁定的 Class A 变体。"""
        return self.is_class_a() and self.skill_lock_enabled


from core.paths import SUMMONS_DIR


class SummonDefinitionRegistry:
    """从 JSON 文件加载并管理召唤物定义。"""

    def __init__(self, directory: Optional[Path] = None):
        if directory is None:
            directory = SUMMONS_DIR
        self.directory = directory
        self._definitions: Dict[str, SummonDefinition] = {}
        self.load()

    def load(self):
        """重新加载目录下所有 *.json 文件。"""
        self._definitions.clear()
        if not self.directory.exists():
            return
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        definition = SummonDefinition(**item)
                        self._definitions[definition.name] = definition
                elif isinstance(data, dict):
                    definition = SummonDefinition(**data)
                    self._definitions[definition.name] = definition
            except Exception:
                # 单个文件损坏不应影响其他定义加载
                continue

    def get(self, name: str) -> Optional[SummonDefinition]:
        return self._definitions.get(name)

    def all(self) -> List[SummonDefinition]:
        return list(self._definitions.values())

    def names(self) -> List[str]:
        return sorted(self._definitions.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._definitions

    def __len__(self) -> int:
        return len(self._definitions)

    def find_by_alias(self, name: str) -> Optional[SummonDefinition]:
        """根据干员名或别名查找召唤物定义。

        匹配规则：
          1. 先精确匹配 name；
          2. 再匹配 aliases（归一化后比较）；
          3. 多个别名命中时优先返回 is_default_alias=True 的项。
        """
        if not name:
            return None
        normalized = self._normalize(name)

        matches = []
        for definition in self._definitions.values():
            if normalized == self._normalize(definition.name):
                matches.append(definition)
                continue
            for alias in definition.aliases:
                if normalized == self._normalize(alias):
                    matches.append(definition)
                    break

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0]
        # 多个命中时优先默认项，否则返回第一个
        for definition in matches:
            if definition.is_default_alias:
                return definition
        return matches[0]

    @staticmethod
    def _normalize(text: str) -> str:
        """归一化用于匹配的文本：去除首尾空白并统一小写。"""
        return text.strip().lower()
