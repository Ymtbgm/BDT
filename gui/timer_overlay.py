from pathlib import Path

from core.base.paths import GUI_TEMPLATE_DIR
from gui._window_effects import set_window_topmost
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
)
from PyQt6.QtCore import Qt, QPoint, QTimer
from PyQt6.QtGui import QFont, QPixmap, QColor


class TimerOverlay(QWidget):
    """悬浮在最上方的半透明计时器窗口，支持手动暂停/继续。"""

    _WING_HEIGHT = 35

    def __init__(self, on_pause_clicked=None, on_reset_clicked=None, on_stop_clicked=None, debug=False, parent=None):
        super().__init__(parent)
        self._drag_pos: QPoint | None = None
        self._debug = debug
        self._on_pause_clicked = on_pause_clicked
        self._on_reset_clicked = on_reset_clicked
        self._on_stop_clicked = on_stop_clicked

        # 置顶通过 Windows SetWindowPos(HWND_TOPMOST) 实现，避免与
        # WA_TranslucentBackground + FramelessWindowHint 组合导致的不透明问题
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # 顶层水平布局：左翼 | 信息面板 | 右翼
        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.left_wing_label = self._create_wing_label("left_wing.png")
        top_layout.addWidget(self.left_wing_label)

        # 信息面板：圆角矩形，内部显示秒帧与 rate/running
        self.info_box = QWidget()
        self.info_box.setStyleSheet(
            "background-color: #fff0f0; "
            "border: 3px solid #ff0000; "
            "border-radius: 20px; "
            "padding: 6px;"
        )
        info_layout = QVBoxLayout(self.info_box)
        info_layout.setContentsMargins(6, 4, 6, 4)
        info_layout.setSpacing(0)

        self.time_label = QLabel("0s 0f")
        self.time_label.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #ff0000; background-color: transparent; border: none;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.time_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        info_layout.addWidget(self.time_label)

        self.info_label = QLabel("rate=1.0" if debug else "")
        self.info_label.setFont(QFont("Consolas", 10))
        self.info_label.setStyleSheet("color: #cc0000; background-color: transparent; border: none;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.info_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.info_label.setVisible(debug)
        info_layout.addWidget(self.info_label)

        top_layout.addWidget(self.info_box)

        self.right_wing_label = self._create_wing_label("right_wing.png")
        top_layout.addWidget(self.right_wing_label)

        layout.addLayout(top_layout)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        btn_layout.addStretch()

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setFixedWidth(60)
        self.btn_pause.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.btn_pause.setFont(QFont("Microsoft YaHei", 10))
        self.btn_pause.clicked.connect(self._handle_pause_click)
        btn_layout.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setFixedWidth(60)
        self.btn_reset.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.btn_reset.setFont(QFont("Microsoft YaHei", 10))
        self.btn_reset.clicked.connect(self._handle_reset_click)
        btn_layout.addWidget(self.btn_reset)

        self.btn_close = QPushButton("关闭")
        self.btn_close.setFixedWidth(60)
        self.btn_close.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.btn_close.setFont(QFont("Microsoft YaHei", 10))
        self.btn_close.clicked.connect(self._handle_close_click)
        btn_layout.addWidget(self.btn_close)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.info_box.setFixedWidth(170)
        height = 130 if debug else 100
        self.setFixedSize(320, height)
        self.move(10, 15)

        # Windows 可能会丢失 TOPMOST 状态，定期重新应用
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(lambda: set_window_topmost(self))
        self._topmost_timer.start(500)

    def _create_wing_label(self, filename: str) -> QLabel:
        """加载翅膀图片（优先使用已裁剪版本），缩放后返回 QLabel。"""
        label = QLabel()
        label.setStyleSheet("background-color: transparent; border: none;")
        resource_dir = GUI_TEMPLATE_DIR
        # 优先使用裁剪掉透明边距的版本
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
        return label

    def _handle_pause_click(self):
        if self._on_pause_clicked:
            self._on_pause_clicked()

    def _handle_reset_click(self):
        if self._on_reset_clicked:
            self._on_reset_clicked()

    def _handle_close_click(self):
        if self._on_stop_clicked:
            self._on_stop_clicked()

    def update_time(self, elapsed_ms: float, seconds: int, frame: int, rate: float, paused: bool):
        self.time_label.setText(f"{seconds}s {frame:02d}f")
        if self._debug:
            state = "paused" if paused else "running"
            self.info_label.setText(f"{elapsed_ms:.0f}ms  rate={rate}  {state}")

    def set_pause_text(self, paused: bool):
        self.btn_pause.setText("继续" if paused else "暂停")

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

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor
        painter = QPainter(self)
        # Source 模式：用完全透明色直接替换目标像素
        painter.setCompositionMode(QPainter.CompositionMode(1))
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))


