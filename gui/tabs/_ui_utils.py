"""Tab 界面通用 UI 工具函数。"""

import os
import tempfile

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import QGroupBox, QStyle, QStyleOptionGroupBox

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


class IconGroupBox(QGroupBox):
    """带图标的 QGroupBox，使用 QIcon + QPainter 直接绘制图标。

    相比 QSS background-image，这种方式和 QTabWidget.setTabIcon 走同一条
    渲染路径，能正确处理高 DPI 缩放，避免样式引擎二次拉伸导致的模糊。
    """

    def __init__(self, title: str, icon_filename: str, icon_size: int = 20, parent=None):
        super().__init__(title, parent)
        self._icon_filename = icon_filename
        self._icon_size = icon_size
        self._icon: QIcon | None = None
        self._load_icon()
        self._setup_style()

    def _load_icon(self):
        icon_path = str(gui_template("ui_icons") / self._icon_filename)
        pixmap = QPixmap(icon_path)
        if not pixmap.isNull():
            self._icon = QIcon(pixmap)
        else:
            self._icon = None

    def _setup_style(self):
        size = self._icon_size
        self.setStyleSheet(
            f"QGroupBox {{"
            f"  background-color: #fbfbfb;"
            f"  border: 1px solid #dddddd;"
            f"  border-radius: 4px;"
            f"  margin-top: {size + 6}px;"
            f"}}"
            f"QGroupBox::title {{"
            f"  subcontrol-origin: margin;"
            f"  subcontrol-position: top left;"
            f"  left: {size + 14}px;"
            f"  padding-left: 4px;"
            f"  padding-right: 4px;"
            f"  padding-top: 2px;"
            f"  padding-bottom: 2px;"
            f"  background-color: #fbfbfb;"
            f"  font-weight: bold;"
            f"}}"
        )
        font = self.font()
        font.setBold(True)
        self.setFont(font)

    def paintEvent(self, event):
        # 先让父类画出完整的 group box（含标题文字）
        super().paintEvent(event)
        if self._icon is None:
            return

        option = QStyleOptionGroupBox()
        self.initStyleOption(option)
        label_rect = self.style().subControlRect(
            QStyle.ComplexControl.CC_GroupBox,
            option,
            QStyle.SubControl.SC_GroupBoxLabel,
            self,
        )

        # 在标题文字左侧绘制图标，像 tab 图标一样使用 QIcon.paint
        painter = QPainter(self)
        icon_size = self._icon_size
        x = label_rect.left() - icon_size - 6
        y = label_rect.top() + (label_rect.height() - icon_size) // 2
        self._icon.paint(painter, x, y, icon_size, icon_size)
        painter.end()


def create_icon_group_box(
    title: str, icon_filename: str, icon_size: int = 20, parent=None
) -> IconGroupBox:
    """创建带图标的 IconGroupBox，保持和原 set_group_box_icon 接近的调用方式。"""
    return IconGroupBox(title, icon_filename, icon_size, parent)
