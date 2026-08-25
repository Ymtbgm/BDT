from typing import Any, Dict, List, Optional

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper

from .base import SpecialBehavior
from .config_field import ConfigField
from .probability_checkpoints import get_method_registry


class ProbabilityCheckpointBehavior(SpecialBehavior):
    """概率点检查入口：根据用户选择的 check_method 分派到具体检查方法。"""

    behavior_id = "概率点检查"
    description = "在指定时间点执行一项概率/状态检查，条件不满足则重新开始关卡，用于凸图/凹概率。"

    # 旧脚本没有 check_method 时的默认方法
    _DEFAULT_METHOD_ID = "grid_target"

    def get_config_fields(self) -> List[ConfigField]:
        registry = get_method_registry()
        methods = registry.list_methods()
        options = [
            {"label": method.description or method.method_id, "value": method.method_id}
            for method in methods
        ]
        return [
            ConfigField(
                name="check_method",
                label="检查方式",
                type="choice",
                default=self._DEFAULT_METHOD_ID,
                options=options,
                hint="选择本次概率点检查的具体方式",
            ),
        ]

    def _get_method(self, params: Optional[Dict[str, Any]]):
        registry = get_method_registry()
        method_id = (params or {}).get("check_method", "")
        if not method_id:
            method_id = self._DEFAULT_METHOD_ID
        return registry.get(method_id)

    def _get_method_params(self, params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """返回排除 check_method 后的方法专属参数。

        同时兼容旧脚本：旧脚本没有 check_method，直接把全部 params 传给默认方法。
        """
        params = params or {}
        result = dict(params)
        result.pop("check_method", None)
        return result

    def execute(
        self,
        capture: WindowCapture,
        grid_mapper: GridMapper,
        params: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        method = self._get_method(params)
        if method is None:
            method_id = (params or {}).get("check_method", "")
            print(f"[概率点检查] 未找到检查方法: {method_id}，视为通过")
            return True

        method_params = self._get_method_params(params)
        return method.execute(capture, grid_mapper, method_params, context)
