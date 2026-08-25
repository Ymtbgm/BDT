import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional

from .base import SpecialBehavior


class SpecialBehaviorRegistry:
    """特殊行为注册表，自动发现并管理所有 SpecialBehavior 子类。"""

    def __init__(self):
        self._behaviors: Dict[str, SpecialBehavior] = {}

    def register(self, behavior: SpecialBehavior):
        """注册一个行为实例。"""
        bid = behavior.behavior_id
        if not bid:
            raise ValueError("behavior_id 不能为空")
        self._behaviors[bid] = behavior

    def get(self, behavior_id: str) -> Optional[SpecialBehavior]:
        """根据 ID 获取行为实例。"""
        return self._behaviors.get(behavior_id)

    def list_behaviors(self) -> List[SpecialBehavior]:
        """返回所有已注册行为，按 behavior_id 排序。"""
        return [self._behaviors[k] for k in sorted(self._behaviors.keys())]

    def discover(self):
        """自动扫描 core.special_behaviors 包下所有模块并注册行为。"""
        from . import base as _base_module

        package = __import__(__name__.rsplit(".", 1)[0], fromlist=[""])
        for _, module_name, is_pkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            if is_pkg:
                continue
            if module_name.endswith(".base") or module_name.endswith(".registry") or module_name.endswith(".config_field"):
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"[特殊行为] 加载模块 {module_name} 失败: {e}")
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, SpecialBehavior)
                    and obj is not SpecialBehavior
                    and obj.behavior_id
                ):
                    try:
                        self.register(obj())
                    except Exception as e:
                        print(f"[特殊行为] 注册 {obj.__name__} 失败: {e}")


# 全局单例
_REGISTRY: Optional[SpecialBehaviorRegistry] = None


def get_registry() -> SpecialBehaviorRegistry:
    """获取全局注册表，首次调用时自动发现行为。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SpecialBehaviorRegistry()
        _REGISTRY.discover()
    return _REGISTRY
