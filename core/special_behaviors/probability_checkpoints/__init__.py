"""概率点检查子方法包。

每个检查方式是一个继承自 ProbabilityCheckpointMethod 的类，
ProbabilityCheckpointBehavior 会根据 params["check_method"] 自动分派。
"""

from .base import ProbabilityCheckpointMethod
from .registry import get_method_registry

__all__ = ["ProbabilityCheckpointMethod", "get_method_registry"]
