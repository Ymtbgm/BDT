from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QFont


class TimerOverlay(QWidget):
    """悬浮在最上方的半透明计时器窗口，支持手动暂停/继续。"""

    def __init__(self, on_pause_clicked=None, on_reset_clicked=None, parent=None):
        super().__init__(parent)
        self._drag_pos: QPoint | None = None
        self._on_pause_clicked = on_pause_clicked
        self._on_reset_clicked = on_reset_clicked

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(
            "background-color: rgba(0, 0, 0, 180); "
            "border-radius: 8px; "
            "padding: 4px;"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.time_label = QLabel("0s 0f")
        self.time_label.setFont(QFont("Consolas", 20, QFont.Weight.Bold))
        self.time_label.setStyleSheet("color: #00ff00;")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        self.info_label = QLabel("rate=1.0  running")
        self.info_label.setFont(QFont("Consolas", 10))
        self.info_label.setStyleSheet("color: #cccccc;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_pause = QPushButton("暂停")
        self.btn_pause.setStyleSheet(
            "background-color: rgba(255, 255, 255, 50); "
            "color: white; "
            "border: 1px solid rgba(255, 255, 255, 100); "
            "border-radius: 4px; "
            "padding: 2px;"
        )
        self.btn_pause.setFont(QFont("Microsoft YaHei", 10))
        self.btn_pause.clicked.connect(self._handle_pause_click)
        btn_layout.addWidget(self.btn_pause)

        self.btn_reset = QPushButton("重置")
        self.btn_reset.setStyleSheet(
            "background-color: rgba(255, 255, 255, 50); "
            "color: white; "
            "border: 1px solid rgba(255, 255, 255, 100); "
            "border-radius: 4px; "
            "padding: 2px;"
        )
        self.btn_reset.setFont(QFont("Microsoft YaHei", 10))
        self.btn_reset.clicked.connect(self._handle_reset_click)
        btn_layout.addWidget(self.btn_reset)

        layout.addLayout(btn_layout)

        self.setFixedSize(180, 115)
        self.move(100, 100)

    def _handle_pause_click(self):
        if self._on_pause_clicked:
            self._on_pause_clicked()

    def _handle_reset_click(self):
        if self._on_reset_clicked:
            self._on_reset_clicked()

    def update_time(self, elapsed_ms: float, seconds: int, frame: int, rate: float, paused: bool):
        self.time_label.setText(f"{seconds}s {frame:02d}f")
        state = "paused" if paused else "running"
        self.info_label.setText(f"{elapsed_ms:6.0f}ms  rate={rate}  {state}")

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
