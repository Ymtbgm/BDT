from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper

from core.special_behaviors.config_field import ConfigField


class ProbabilityCheckpointMethod(ABC):
    """概率点检查具体方法的抽象基类。"""

    method_id: str = ""
    description: str = ""

    @abstractmethod
    def get_config_fields(self) -> List[ConfigField]:
        """返回该方法需要用户填写的配置项。"""
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        capture: WindowCapture,
        grid_mapper: GridMapper,
        params: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """执行检查。

        Args:
            capture: 窗口截图对象
            grid_mapper: 地图格子映射器
            params: 用户配置参数（已包含 check_method）
            context: 执行上下文

        Returns:
            True: 条件满足，继续执行脚本
            False: 条件不满足，触发重试
        """
        raise NotImplementedError
