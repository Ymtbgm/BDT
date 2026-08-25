from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from core.capture.capture import WindowCapture
from core.map.grid_mapper import GridMapper

from .config_field import ConfigField


class SpecialBehavior(ABC):
    """特殊行为基类。

    每个特殊行为需要实现：
    - behavior_id: 唯一标识
    - description: 前端展示的描述
    - get_config_fields: 前端配置项列表
    - execute: 执行检查/操作，返回 True 表示通过，False 表示需要重试
    """

    behavior_id: str = ""
    description: str = ""

    @abstractmethod
    def get_config_fields(self) -> List[ConfigField]:
        """返回该行为需要用户填写的配置项。"""
        raise NotImplementedError

    @abstractmethod
    def execute(
        self,
        capture: WindowCapture,
        grid_mapper: GridMapper,
        params: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
    ) -> bool:
        """执行行为/检查。

        Args:
            capture: 窗口截图对象
            grid_mapper: 地图格子映射器
            params: 用户配置参数
            context: 执行上下文，可包含 executor、pool 等对象供扩展使用

        Returns:
            True: 条件满足，继续执行脚本
            False: 条件不满足，触发重试
        """
        raise NotImplementedError
