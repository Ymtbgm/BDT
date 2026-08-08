from pathlib import Path

from core.base.paths import GUI_TEMPLATE_DIR
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QGraphicsDropShadowEffect,
    QSizePolicy,
)
from PyQt6.QtGui import QFont, QPixmap, QColor


class InfoCollectionOverlay(QWidget):
    """信息录入/录制提示浮窗：无边框、置顶、可显示阶段文本或计时时间。"""

    _WING_HEIGHT = 35

    def __init__(self, debug=False, parent=None):
        super().__init__(parent)
        self._drag_pos: QPoint | None = None
        self._debug = debug
        self._timer_mode = False

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.left_wing_label = self._create_wing_label("left_wing.png")
        top_layout.addWidget(self.left_wing_label)

        self.info_box = QWidget()
        self.info_box.setStyleSheet(
            "background-color: qradialgradient("
            "cx:0.5, cy:0.5, radius:0.75, fx:0.5, fy:0.5, "
            "stop:0 rgba(252, 252, 252, 245), "
            "stop:0.75 rgba(255, 245, 245, 245), "
            "stop:1 rgba(255, 225, 225, 245)); "
            "border: 3px solid #ff0000; "
            "border-radius: 20px; "
            "padding: 6px;"
        )
        shadow = QGraphicsDropShadowEffect(self.info_box)
        shadow.setBlurRadius(16)
        shadow.setColor(QColor(255, 0, 0, 100))
        shadow.setOffset(0, 4)
        self.info_box.setGraphicsEffect(shadow)

        info_layout = QVBoxLayout(self.info_box)
        info_layout.setContentsMargins(6, 4, 6, 4)
        info_layout.setSpacing(0)

        self.main_label = QLabel("信息录入准备中...")
        self.main_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.main_label.setStyleSheet(
            "color: #cc0000; background-color: transparent; border: none;"
        )
        self.main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.main_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        info_layout.addWidget(self.main_label)

        self.debug_label = QLabel("")
        self.debug_label.setFont(QFont("Consolas", 10))
        self.debug_label.setStyleSheet(
            "color: #cc0000; background-color: transparent; border: none;"
        )
        self.debug_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.debug_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        self.debug_label.setVisible(debug)
        info_layout.addWidget(self.debug_label)

        top_layout.addWidget(self.info_box)

        self.right_wing_label = self._create_wing_label("right_wing.png")
        top_layout.addWidget(self.right_wing_label)

        layout.addLayout(top_layout)

        self.info_box.setFixedWidth(200)
        self.setFixedSize(320, 90)
        self.move(20, 20)

    def _create_wing_label(self, filename: str) -> QLabel:
        """加载翅膀图片（优先使用已裁剪版本），缩放后返回 QLabel。"""
        label = QLabel()
        label.setStyleSheet("background-color: transparent; border: none;")
        resource_dir = GUI_TEMPLATE_DIR
        cropped_path = resource_dir / f"{Path(filename).stem}_cropped.png"
        path = cropped_path if cropped_path.exists() else resource_dir / filename
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            return label
        scaled = pixmap.scaledToHeight(
            self._WING_HEIGHT,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(scaled)
        label.setFixedSize(scaled.size())
        wing_shadow = QGraphicsDropShadowEffect(label)
        wing_shadow.setBlurRadius(12)
        wing_shadow.setColor(QColor(255, 0, 0, 90))
        wing_shadow.setOffset(0, 3)
        label.setGraphicsEffect(wing_shadow)
        return label

    def _set_phase_font(self):
        self.main_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.main_label.setStyleSheet(
            "color: #cc0000; background-color: transparent; border: none;"
        )

    def _set_timer_font(self):
        self.main_label.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        self.main_label.setStyleSheet(
            "color: #ff0000; background-color: transparent; border: none;"
        )

    def set_phase(self, phase: str):
        """更新当前阶段文本。"""
        if self._timer_mode:
            self._timer_mode = False
            self._set_phase_font()
            self.debug_label.setVisible(False)
        self.main_label.setText(phase)
        self.update()

    def set_time(self, seconds: int, frame: int, elapsed_ms: float = 0.0,
                 rate: float = 1.0, paused: bool = False):
        """切换到计时显示模式并更新时间。"""
        if self._debug:
            print(f"[InfoCollectionOverlay.set_time] timer_mode={self._timer_mode} seconds={seconds} frame={frame}")
        if not self._timer_mode:
            self._timer_mode = True
            self._set_timer_font()
            self.debug_label.setVisible(self._debug)
        self.main_label.setText(f"{seconds}s {frame:02d}f")
        if self._debug:
            state = "paused" if paused else "running"
            self.debug_label.setText(f"{elapsed_ms:.0f}ms  rate={rate}  {state}")
        self.update()

    def close_overlay(self):
        """关闭浮窗。"""
        self.close()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_pos is not None and event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        self._drag_pos = None
        event.accept()
