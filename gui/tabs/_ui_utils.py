"""Tab 界面通用 UI 工具函数。"""

import os
import tempfile

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGroupBox

from core.base.paths import gui_template


def scaled_icon_path(filename: str, size: int = 20) -> str:
    """加载 ui_icons 图标并缩放到指定大小，返回临时 PNG 路径供 QSS 使用。"""
    src_path = str(gui_template("ui_icons") / filename)
    cache_dir = os.path.join(tempfile.gettempdir(), "bdt_gui_icons")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{filename}_{size}.png")
    if os.path.exists(cache_path):
        return cache_path
    pixmap = QPixmap(src_path)
    if pixmap.isNull():
        return src_path
    scaled = pixmap.scaled(
        size,
        size,
        aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio,
        transformMode=Qt.TransformationMode.SmoothTransformation,
    )
    scaled.save(cache_path, "PNG")
    return cache_path


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
        "}"
    )


def set_group_box_icon(group: QGroupBox, filename: str, size: int = 20):
    """在 QGroupBox 标题左侧添加图标，并将整个 box 设为浅灰色以区分主背景。"""
    icon_path = scaled_icon_path(filename, size).replace("\\", "/")
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
        f"}}"
    )
