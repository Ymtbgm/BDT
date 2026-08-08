"""右下角自动消失提示（Toast）。"""

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QApplication, QGraphicsOpacityEffect, QSizePolicy


class Toast(QWidget):
    """在屏幕右下角显示一条自动消失的提示。

    使用示例：
        Toast.show_message(parent, "更新完成", Toast.Type.SUCCESS)
    """

    class Type:
        INFO = "info"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"

    DEFAULT_DURATION_MS = 5000

    _TYPE_COLORS = {
        Type.INFO: "#6495ED",
        Type.SUCCESS: "#50C878",
        Type.WARNING: "#FFC107",
        Type.ERROR: "#FF6B6B",
    }

    def __init__(
        self,
        parent=None,
        message: str = "",
        toast_type: str = Type.INFO,
        duration_ms: int = DEFAULT_DURATION_MS,
    ):
        super().__init__(parent)
        self._toast_type = toast_type
        self._duration_ms = duration_ms

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.NoDropShadowWindowHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._setup_ui(message)
        self._setup_animation()
        self._position_on_screen()

    def _setup_ui(self, message: str):
        dot_color = self._TYPE_COLORS.get(self._toast_type, self._TYPE_COLORS[self.Type.INFO])

        self.setStyleSheet("""
            QWidget#ToastContainer {
                background-color: #FFFFFF;
                border: 1px solid #E0E0E0;
                border-radius: 10px;
            }
            QLabel#ToastMessage {
                color: #333333;
                font-size: 13px;
                background: transparent;
                border: none;
            }
        """)

        container = QWidget(self)
        container.setObjectName("ToastContainer")

        layout = QHBoxLayout(container)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        # 彩色圆点指示器
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"""
            background-color: {dot_color};
            border-radius: 5px;
        """)
        layout.addWidget(dot)

        self._label = QLabel(message)
        self._label.setObjectName("ToastMessage")
        self._label.setFont(QFont("Microsoft YaHei", 10))
        self._label.setWordWrap(True)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout.addWidget(self._label)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(container)

        self.setMinimumWidth(240)
        self.setMaximumWidth(400)
        self.adjustSize()

    def _setup_animation(self):
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(350)
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_animation.finished.connect(self.close)

        QTimer.singleShot(self._duration_ms, self._start_fade)

    def _start_fade(self):
        self._fade_animation.start()

    def _position_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 24
        x = screen.right() - self.width() - margin
        y = screen.bottom() - self.height() - margin
        self.move(QPoint(x, y))

    def showEvent(self, event):
        super().showEvent(event)
        self._position_on_screen()

    @classmethod
    def show_message(
        cls,
        parent=None,
        message: str = "",
        toast_type: str = Type.INFO,
        duration_ms: int = DEFAULT_DURATION_MS,
    ):
        """创建并显示一条 Toast。"""
        toast = cls(parent, message, toast_type, duration_ms)
        toast.show()
        return toast
