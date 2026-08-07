"""轻量日志开关工具。

默认关闭模型加载、warm-up 等非关键输出，避免执行器/解析器启动时刷屏。
如需调试，设置环境变量 ``ARK_VERBOSE=1`` 或在代码里调用 ``set_verbose(True)`` 即可重新显示。
"""

import os
import sys
from typing import Any

_VERBOSE = os.environ.get("ARK_VERBOSE", "0").lower() in ("1", "true", "yes")
VERBOSE = _VERBOSE


def set_verbose(value: bool) -> None:
    """运行时覆盖 verbose 开关。"""
    global _VERBOSE, VERBOSE
    _VERBOSE = bool(value)
    VERBOSE = _VERBOSE


def is_verbose() -> bool:
    """是否启用详细输出。"""
    return _VERBOSE


def log_info(message: Any) -> None:
    """在 verbose 模式下输出到 stdout。"""
    if _VERBOSE:
        print(message)


def log_error(message: Any) -> None:
    """始终输出到 stderr，用于真正的错误/失败信息。"""
    print(message, file=sys.stderr)
