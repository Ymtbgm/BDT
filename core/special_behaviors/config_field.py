from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConfigField:
    """描述一个特殊行为在前端需要用户填写的配置项。"""

    name: str
    label: str
    type: str = "str"  # 支持: str, int, choice, bool, unit
    default: Any = None
    options: List[Dict[str, str]] = field(default_factory=list)
    hint: str = ""

    def __post_init__(self):
        if self.type == "choice" and not self.options:
            raise ValueError(f"配置项 {self.name} 类型为 choice 时必须提供 options")
