"""右下角自动消失提示（Toast），支持多个 Toast 堆叠显示。"""

from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QApplication, QGraphicsOpacityEffect, QSizePolicy


class Toast(QWidget):
    """在屏幕右下角显示一条自动消失的提示，多个 Toast 会自动向上堆叠。

    使用示例：
        Toast.show_message(parent, "更新完成", Toast.Type.SUCCESS)
    """

    class Type:
        INFO = "info"
        SUCCESS = "success"
        WARNING = "warning"
        ERROR = "error"

    DEFAULT_DURATION_MS = 5000
    MARGIN = 24
    GAP = 12

    _TYPE_COLORS = {
        Type.INFO: "#6495ED",
        Type.SUCCESS: "#50C878",
        Type.WARNING: "#FFC107",
        Type.ERROR: "#FF6B6B",
    }

    _active_toasts: list["Toast"] = []

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
        # 开始淡出时从堆叠列表移除，并重新排列剩余 Toast
        if self in Toast._active_toasts:
            Toast._active_toasts.remove(self)
            Toast._reposition_all()
        self._fade_animation.start()

    def _position_for_index(self, index: int) -> QPoint:
        """计算指定索引 Toast 的左上角坐标（从底部向上堆叠）。"""
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.right() - self.width() - self.MARGIN
        y = screen.bottom() - self.height() - self.MARGIN
        # index 0 是最下面，向上递增
        for i in range(index):
            if i < len(Toast._active_toasts):
                y -= Toast._active_toasts[i].height() + self.GAP
        return QPoint(x, y)

    def _reposition(self):
        """根据当前在列表中的位置移动自己。"""
        if self in Toast._active_toasts:
            idx = Toast._active_toasts.index(self)
            self.move(self._position_for_index(idx))

    @classmethod
    def _reposition_all(cls):
        """重新排列所有激活的 Toast。"""
        for toast in cls._active_toasts:
            toast._reposition()

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition()

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
        cls._active_toasts.append(toast)
        cls._reposition_all()
        toast.show()
        return toast
