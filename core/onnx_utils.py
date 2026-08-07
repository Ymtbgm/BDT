"""ONNX Runtime 共享工具。

集中管理 execution provider 选择逻辑与 session 选项，避免 avatar_matcher、
digit_recognizer 等模块各自实现，同时统一关闭 C 层 INFO 日志，防止在执行器
/录制器启动时刷屏或阻塞 QProcess 管道。
"""

import os
from typing import Any, List, Optional

# 默认只保留 ERROR 及以上级别的 ONNX Runtime C 层日志，避免模型加载时
# 输出大量 "Removing initializer ..." 等信息并可能导致 QProcess 管道阻塞。
os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")


def get_onnx_providers(
    prefer: Optional[str] = None,
    allow_cuda: bool = False,
) -> List[str]:
    """返回可用的 ONNX Runtime execution provider 列表。

    默认优先 DirectML，否则 CPU。除非显式允许，否则绝不启用 CUDA。

    Args:
        prefer: 可选 "directml" / "cpu"；None 表示自动选择。
        allow_cuda: 是否允许 CUDA provider（默认 False，遵守项目无 CUDA 偏好）。
    """
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime 未安装") from exc

    available = set(ort.get_available_providers())
    providers: List[str] = []

    if not allow_cuda and "CUDAExecutionProvider" in available:
        available.discard("CUDAExecutionProvider")

    if prefer == "directml" or prefer is None:
        if "DmlExecutionProvider" in available:
            providers.append("DmlExecutionProvider")
    if prefer == "cpu" or prefer is None:
        if "CPUExecutionProvider" in available:
            providers.append("CPUExecutionProvider")

    if not providers and available:
        providers.extend(available)

    return providers


def create_session_options() -> Any:
    """创建统一配置的 ONNX Runtime SessionOptions。

    默认开启图优化、mem pattern，并将日志级别设为 WARNING，避免加载模型时
    输出海量 C 层日志。
    """
    import onnxruntime as ort

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_mem_pattern = True
    sess_options.log_severity_level = 3  # ERROR
    return sess_options
