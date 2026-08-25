"""特殊行为插件包。

本包用于存放按时间执行的脚本特殊行为。每个行为是一个独立模块，内部定义一个
继承自 SpecialBehavior 的类；registry 会在导入时自动发现并注册。
"""

from .base import SpecialBehavior
from .config_field import ConfigField
from .registry import SpecialBehaviorRegistry, get_registry

__all__ = ["SpecialBehavior", "ConfigField", "SpecialBehaviorRegistry", "get_registry"]
