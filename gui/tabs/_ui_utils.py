"""Tab 界面通用 UI 工具函数。"""

import os
import tempfile

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import QGroupBox

from core.base.paths import gui_template


def scaled_icon_path(filename: str, size: int = 20, fast: bool = False) -> str:
    """加载 ui_icons 图标并缩放到指定大小，返回临时 PNG 路径供 QSS 使用。

    fast=True 时使用最近邻缩放（FastTransformation），适合线条简单的图标，
    能避免 SmoothTransformation 带来的边缘柔化/模糊。
    """
    src_path = str(gui_template("ui_icons") / filename)
    cache_dir = os.path.join(tempfile.gettempdir(), "bdt_gui_icons")
    os.makedirs(cache_dir, exist_ok=True)
    mode_key = "fast" if fast else "smooth"
    cache_path = os.path.join(cache_dir, f"{filename}_{size}_{mode_key}.png")
    if os.path.exists(cache_path):
        return cache_path
    pixmap = QPixmap(src_path)
    if pixmap.isNull():
        return src_path
    transform_mode = (
        Qt.TransformationMode.FastTransformation
        if fast
        else Qt.TransformationMode.SmoothTransformation
    )
    scaled = pixmap.scaled(
        size,
        size,
        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
        transformMode=transform_mode,
    )
    scaled.save(cache_path, "PNG")
    return cache_path


def _set_group_box_bold_title(group: QGroupBox):
    """将 QGroupBox 标题字体设为粗体（QSS 在某些样式下不生效，直接设置字体更可靠）。"""
    font = group.font()
    font.setBold(True)
    group.setFont(font)


def set_group_box_style(group: QGroupBox):
    """将 QGroupBox 设为浅灰色背景（不带图标）。"""
    group.setStyleSheet(
        "QGroupBox {"
        "  background-color: #fbfbfb;"
        "  border: 1px solid #dddddd;"
        "  border-radius: 4px;"
        "  margin-top: 8px;"
        "}"
        "QGroupBox::title {"
        "  subcontrol-origin: margin;"
        "  subcontrol-position: top left;"
        "  left: 8px;"
        "  padding-left: 4px;"
        "  padding-right: 4px;"
        "  padding-top: 2px;"
        "  padding-bottom: 2px;"
        "  background-color: #fbfbfb;"
        "  font-weight: bold;"
        "}"
    )
    _set_group_box_bold_title(group)


def set_group_box_icon(group: QGroupBox, filename: str, size: int = 20):
    """在 QGroupBox 标题左侧添加图标，并将整个 box 设为浅灰色以区分主背景。"""
    icon_path = scaled_icon_path(filename, size, fast=True).replace("\\", "/")
    group.setStyleSheet(
        f"QGroupBox {{"
        f"  background-color: #fbfbfb;"
        f"  border: 1px solid #dddddd;"
        f"  border-radius: 4px;"
        f"  margin-top: {size + 6}px;"
        f"}}"
        f"QGroupBox::title {{"
        f"  subcontrol-origin: margin;"
        f"  subcontrol-position: top left;"
        f"  left: 8px;"
        f"  padding-left: {size + 6}px;"
        f"  padding-right: 4px;"
        f"  padding-top: 2px;"
        f"  padding-bottom: 2px;"
        f"  background-color: #fbfbfb;"
        f"  background-image: url({icon_path});"
        f"  background-repeat: no-repeat;"
        f"  background-position: left center;"
        f"  font-weight: bold;"
        f"}}"
    )
    _set_group_box_bold_title(group)
