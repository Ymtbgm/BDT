from pathlib import Path

from core.base.paths import GUI_TEMPLATE_DIR
from gui._window_effects import set_window_topmost
from PyQt6.QtCore import Qt, QPoint, pyqtSlot, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSizePolicy, QPushButton,
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

        # 置顶通过 Windows SetWindowPos(HWND_TOPMOST) 实现，避免与
        # WA_TranslucentBackground + FramelessWindowHint 组合导致的不透明问题
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 3)
        layout.setSpacing(2)

        top_layout = QHBoxLayout()
        top_layout.setSpacing(6)
        top_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.left_wing_label = self._create_wing_label("left_wing.png")
        top_layout.addWidget(self.left_wing_label)

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

        # 装载脚本状态小标签（位于计时器主体上方，风格与录制按钮一致）
        self.script_tab = QLabel("当前未装载脚本")
        self.script_tab.setFont(QFont("Microsoft YaHei", 9, QFont.Weight.Bold))
        self.script_tab.setStyleSheet(
            "color: #ff0000; "
            "background-color: rgba(255, 255, 255, 255); "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px 8px; "
            "margin-bottom: 4px;"
        )
        self.script_tab.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.script_tab.setSizePolicy(
            QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed
        )

        center_box = QWidget()
        center_layout = QVBoxLayout(center_box)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        center_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(
            self.script_tab, 0,
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom
        )
        center_layout.addWidget(self.info_box)

        top_layout.addWidget(center_box)

        self.right_wing_label = self._create_wing_label("right_wing.png")
        top_layout.addWidget(self.right_wing_label)

        layout.addLayout(top_layout)

        # 录制控制按钮
        self.button_layout = QHBoxLayout()
        self.button_layout.setSpacing(8)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_button = QPushButton("开始录制")
        self.start_button.setFixedWidth(60)
        self.start_button.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.start_button.setFont(QFont("Microsoft YaHei", 10))
        self.start_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.stop_button = QPushButton("结束录制")
        self.stop_button.setFixedWidth(60)
        self.stop_button.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.stop_button.setFont(QFont("Microsoft YaHei", 10))
        self.stop_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.takeover_button = QPushButton("手动接管")
        self.takeover_button.setFixedWidth(70)
        self.takeover_button.setStyleSheet(
            "background-color: rgba(255, 255, 255, 255); "
            "color: #ff0000; "
            "border: 1px solid #ff0000; "
            "border-radius: 6px; "
            "padding: 2px;"
        )
        self.takeover_button.setFont(QFont("Microsoft YaHei", 10))
        self.takeover_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.takeover_button.setVisible(False)
        self.button_layout.addWidget(self.start_button)
        self.button_layout.addWidget(self.stop_button)
        self.button_layout.addWidget(self.takeover_button)
        layout.addLayout(self.button_layout)
        layout.addStretch(1)

        self.info_box.setFixedWidth(200)
        self.info_box.setMinimumHeight(46)
        self.setFixedSize(320, 140)
        self.move(20, 20)

        # Windows 可能会丢失 TOPMOST 状态，定期重新应用
        self._topmost_timer = QTimer(self)
        self._topmost_timer.timeout.connect(lambda: set_window_topmost(self))
        self._topmost_timer.start(500)

        self._start_callback = None
        self._stop_callback = None
        self._takeover_callback = None
        self.start_button.clicked.connect(self._on_start_clicked)
        self.stop_button.clicked.connect(self._on_stop_clicked)
        self.takeover_button.clicked.connect(self._on_takeover_clicked)
        self.set_recording_state(False)

    def _on_start_clicked(self):
        if self._start_callback is not None:
            self._start_callback()

    def _on_stop_clicked(self):
        if self._stop_callback is not None:
            self._stop_callback()

    def _on_takeover_clicked(self):
        if self._takeover_callback is not None:
            self._takeover_callback()

    def set_button_callbacks(self, start_callback=None, stop_callback=None, takeover_callback=None):
        """设置开始/结束录制按钮和手动接管按钮的回调。"""
        self._start_callback = start_callback
        self._stop_callback = stop_callback
        self._takeover_callback = takeover_callback

    def set_recording_state(self, is_recording: bool):
        """根据是否录制中更新按钮可用状态。"""
        self.start_button.setEnabled(not is_recording)
        self.stop_button.setEnabled(is_recording)

    def set_script_status(self, text: str):
        """更新装载脚本状态文本（顶部小标签）。"""
        self.script_tab.setText(text)
        self.update()

    @pyqtSlot(bool)
    def show_takeover_mode(self, show: bool = True):
        """切换为手动接管模式（隐藏开始/结束，显示手动接管）。"""
        self.start_button.setVisible(not show)
        self.stop_button.setVisible(not show)
        self.takeover_button.setVisible(show)
        self.update()

    def show_record_buttons(self):
        """恢复显示开始/结束录制按钮。"""
        self.show_takeover_mode(False)

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
        return label

    def _set_phase_font(self):
        self.main_label.setFont(QFont("Microsoft YaHei", 12, QFont.Weight.Bold))
        self.main_label.setStyleSheet(
            "color: #cc0000; background-color: transparent; border: none;"
        )

    def _set_save_font(self):
        self.main_label.setFont(QFont("Microsoft YaHei", 9))
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

    def set_save_path(self, path: Path | str):
        """以较小字体显示脚本保存文件名。"""
        if self._timer_mode:
            self._timer_mode = False
        self._set_save_font()
        self.debug_label.setVisible(False)
        name = path.name if isinstance(path, Path) else Path(path).name
        self.main_label.setText(f"保存脚本到：{name}")
        self.update()

    @pyqtSlot(int, int, float, float, bool)
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

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QColor
        painter = QPainter(self)
        # Source 模式：用完全透明色直接替换目标像素
        painter.setCompositionMode(QPainter.CompositionMode(1))
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))


