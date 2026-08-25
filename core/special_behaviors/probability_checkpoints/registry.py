import importlib
import inspect
import pkgutil
from typing import Dict, List, Optional

from .base import ProbabilityCheckpointMethod


class ProbabilityCheckpointMethodRegistry:
    """概率点检查子方法注册表，自动发现并管理所有方法类。"""

    def __init__(self):
        self._methods: Dict[str, ProbabilityCheckpointMethod] = {}

    def register(self, method: ProbabilityCheckpointMethod):
        """注册一个方法实例。"""
        mid = method.method_id
        if not mid:
            raise ValueError("method_id 不能为空")
        self._methods[mid] = method

    def get(self, method_id: str) -> Optional[ProbabilityCheckpointMethod]:
        """根据 ID 获取方法实例。"""
        return self._methods.get(method_id)

    def list_methods(self) -> List[ProbabilityCheckpointMethod]:
        """返回所有已注册方法，按 method_id 排序。"""
        return [self._methods[k] for k in sorted(self._methods.keys())]

    def discover(self):
        """自动扫描 probability_checkpoints 包下所有模块并注册方法。"""
        from . import base as _base_module

        package = __import__(__name__.rsplit(".", 1)[0], fromlist=[""])
        for _, module_name, is_pkg in pkgutil.iter_modules(
            package.__path__, package.__name__ + "."
        ):
            if is_pkg:
                continue
            if module_name.endswith(".base") or module_name.endswith(".registry"):
                continue
            try:
                module = importlib.import_module(module_name)
            except Exception as e:
                print(f"[概率点检查方法] 加载模块 {module_name} 失败: {e}")
                continue
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    issubclass(obj, ProbabilityCheckpointMethod)
                    and obj is not ProbabilityCheckpointMethod
                    and obj.method_id
                ):
                    try:
                        self.register(obj())
                    except Exception as e:
                        print(f"[概率点检查方法] 注册 {obj.__name__} 失败: {e}")


_REGISTRY: Optional[ProbabilityCheckpointMethodRegistry] = None


def get_method_registry() -> ProbabilityCheckpointMethodRegistry:
    """获取全局单例注册表，首次调用时自动发现方法。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = ProbabilityCheckpointMethodRegistry()
        _REGISTRY.discover()
    return _REGISTRY
