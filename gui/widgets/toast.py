"""右下角自动消失提示（Toast）。"""

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont, QPalette, QColor
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication, QGraphicsOpacityEffect


class Toast(QWidget):
    """在屏幕右下角显示一条自动消失的提示。

    使用示例：
        Toast.show_message(parent, "更新完成")
    """

    DEFAULT_DURATION_MS = 5000

    def __init__(self, parent=None, message: str = "", duration_ms: int = DEFAULT_DURATION_MS):
        super().__init__(parent)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._duration_ms = duration_ms
        self._setup_ui(message)
        self._setup_animation()
        self._position_on_screen()

    def _setup_ui(self, message: str):
        self.setStyleSheet("""
            QWidget {
                background-color: rgba(45, 45, 48, 230);
                border-radius: 8px;
            }
            QLabel {
                color: white;
                font-size: 13px;
                padding: 12px 16px;
            }
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(message)
        self._label.setFont(QFont("Microsoft YaHei", 10))
        self._label.setWordWrap(True)
        layout.addWidget(self._label)

        self.setMinimumWidth(200)
        self.setMaximumWidth(360)
        self.adjustSize()

    def _setup_animation(self):
        self._opacity_effect = QGraphicsOpacityEffect(self)
        self._opacity_effect.setOpacity(1.0)
        self.setGraphicsEffect(self._opacity_effect)

        self._fade_animation = QPropertyAnimation(self._opacity_effect, b"opacity")
        self._fade_animation.setDuration(300)
        self._fade_animation.setStartValue(1.0)
        self._fade_animation.setEndValue(0.0)
        self._fade_animation.setEasingCurve(QEasingCurve.Type.InOutQuad)
        self._fade_animation.finished.connect(self.close)

        QTimer.singleShot(self._duration_ms, self._start_fade)

    def _start_fade(self):
        self._fade_animation.start()

    def _position_on_screen(self):
        screen = QApplication.primaryScreen().availableGeometry()
        margin = 20
        x = screen.right() - self.width() - margin
        y = screen.bottom() - self.height() - margin
        self.move(QPoint(x, y))

    def showEvent(self, event):
        super().showEvent(event)
        # 显示后根据实际大小重新定位，避免换行导致位置偏移
        self._position_on_screen()

    @classmethod
    def show_message(cls, parent=None, message: str = "", duration_ms: int = DEFAULT_DURATION_MS):
        """创建并显示一条 Toast。"""
        toast = cls(parent, message, duration_ms)
        toast.show()
        return toast
