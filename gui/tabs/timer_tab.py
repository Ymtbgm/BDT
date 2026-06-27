import action
from core.capture import WindowCapture
from core.region_state_timer import RegionStateTimer
from gui.timer_overlay import TimerOverlay

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QTextEdit, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer


class TimerTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("基于游戏内区域状态的悬浮计时器"))
        layout.addWidget(
            QLabel(
                "暂停区域 : 控制计时/暂停 | "
                "倍速区域 : 控制倍率 1.0x / 0.2x"
            )
        )

        btn_layout = QHBoxLayout()
        self.main_window.btn_timer_start = QPushButton("开始计时")
        self.main_window.btn_timer_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self.main_window.btn_timer_start.clicked.connect(self._start_region_timer)
        btn_layout.addWidget(self.main_window.btn_timer_start)

        self.main_window.btn_timer_stop = QPushButton("停止计时")
        self.main_window.btn_timer_stop.setEnabled(False)
        self.main_window.btn_timer_stop.clicked.connect(self._stop_region_timer)
        btn_layout.addWidget(self.main_window.btn_timer_stop)

        self.main_window.btn_timer_reset = QPushButton("重置时间")
        self.main_window.btn_timer_reset.clicked.connect(self._reset_region_timer)
        btn_layout.addWidget(self.main_window.btn_timer_reset)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.main_window.chk_timer_debug = QCheckBox("Debug 输出")
        layout.addWidget(self.main_window.chk_timer_debug)

        self.main_window.timer_status = QLabel("状态: 就绪")
        layout.addWidget(self.main_window.timer_status)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml(
            """
            <h3>使用说明</h3>
            <ul>
                <li>在编队界面，点击<b>开始计时</b>后，主窗口会自动最小化，并在屏幕左上角显示悬浮计时器。</li>
                <li>计时器窗口默认置顶，可拖动到任意位置，不会被游戏遮挡。</li>
                <li>检测到正式开始游戏(费用条开始动)后自动开始计时。</li>
                <li>采用白像素匹配，<span style="color: red;">因此计时请在1倍速下使用，且不要鼠标操作暂停键，使用快捷键操作。</span></li>
                <li>初始就可能会有几帧误差，如果一直高频切换暂停和恢复会拉大误差。<li>
            </ul>
            """
        )
        layout.addWidget(info)
        layout.addStretch()

    def _build_matchstick_hotkeys(self) -> dict:
        matchstick_cfg = action.get_matchstick_config()
        hotkeys = matchstick_cfg.get("hotkeys", {})
        enabled = matchstick_cfg.get("enabled", {})
        compensation_map = {
            "select_operator": 2.0,
            "pass_166ms": 33.3,
            "pass_50ms": 10.0,
        }
        matchstick_hotkeys = {}
        for name, key in hotkeys.items():
            if enabled.get(name) and name in compensation_map:
                matchstick_hotkeys[name] = {
                    "key": key,
                    "compensation_ms": compensation_map[name],
                }
        return matchstick_hotkeys

    def _start_region_timer(self):
        if self.main_window._region_timer is not None and self.main_window._region_timer.is_running():
            return

        try:
            self.main_window._timer_capture = WindowCapture(backend="mss")
            matchstick_hotkeys = self._build_matchstick_hotkeys()

            self.main_window._region_timer = RegionStateTimer(
                self.main_window._timer_capture,
                pause_key=action.pause_key(),
                debug=self.main_window.chk_timer_debug.isChecked(),
                matchstick_hotkeys=matchstick_hotkeys if matchstick_hotkeys else None,
            )
            self.main_window._region_timer.start(use_cost_detection=True)
        except Exception as e:
            QMessageBox.critical(self.main_window, "启动失败", f"计时器初始化失败:\n{e}")
            return

        self.main_window._timer_overlay = TimerOverlay(
            on_pause_clicked=self._toggle_timer_pause,
            on_reset_clicked=self._reset_region_timer,
        )
        self.main_window._timer_overlay.show()
        self.main_window.showMinimized()

        self.main_window._timer_qtimer = QTimer(self.main_window)
        self.main_window._timer_qtimer.setTimerType(Qt.TimerType.PreciseTimer)
        self.main_window._timer_qtimer.timeout.connect(self._on_timer_tick)
        # 界面刷新保持 20ms，区域 B 采样由 RegionStateTimer 独立线程负责
        self.main_window._timer_qtimer.start(20)

        self.main_window._timer_started = False
        self.main_window.btn_timer_start.setEnabled(False)
        self.main_window.btn_timer_stop.setEnabled(True)
        self.main_window.timer_status.setText("状态: 等待初始状态...")

    def _on_timer_tick(self):
        if self.main_window._region_timer is None:
            return
        info = self.main_window._region_timer.tick()
        if not info.get("started"):
            self.main_window.timer_status.setText(
                f"状态: 等待初始状态 | A={info.get('count_a')} B={info.get('count_b')}"
            )
            return

        elapsed = info["elapsed_ms"]
        s, f = self.main_window._ms_to_sf_for_timer(elapsed)
        self.main_window._timer_overlay.update_time(
            elapsed, s, f, info["rate"], info["paused"]
        )
        self.main_window._timer_overlay.set_pause_text(self.main_window._region_timer.is_manual_paused())
        self.main_window.timer_status.setText(
            f"状态: 运行中 | {s}s {f:02d}f | rate={info['rate']} | paused={info['paused']}"
        )

    def _toggle_timer_pause(self):
        if self.main_window._region_timer is None:
            return
        self.main_window._region_timer.toggle_manual_pause()
        is_paused = self.main_window._region_timer.is_manual_paused()
        if self.main_window._timer_overlay is not None:
            self.main_window._timer_overlay.set_pause_text(is_paused)
        self.main_window.timer_status.setText(f"状态: {'已手动暂停' if is_paused else '运行中'}")

    def _stop_region_timer(self):
        if self.main_window._timer_qtimer is not None:
            self.main_window._timer_qtimer.stop()
            self.main_window._timer_qtimer = None
        if self.main_window._region_timer is not None:
            self.main_window._region_timer.stop()
            self.main_window._region_timer = None
        if self.main_window._timer_overlay is not None:
            self.main_window._timer_overlay.close()
            self.main_window._timer_overlay = None
        self.main_window._timer_capture = None
        self.main_window._timer_started = False
        self.main_window.btn_timer_start.setEnabled(True)
        self.main_window.btn_timer_stop.setEnabled(False)
        self.main_window.timer_status.setText("状态: 已停止")
        if self.main_window.isMinimized():
            self.main_window.showNormal()

    def _reset_region_timer(self):
        if self.main_window._region_timer is None:
            return
        if self.main_window._timer_qtimer is not None:
            self.main_window._timer_qtimer.stop()
        self.main_window._region_timer.start()
        self.main_window._region_timer.manual_pause()
        if self.main_window._timer_overlay is not None:
            self.main_window._timer_overlay.update_time(0.0, 0, 0, 1.0, True)
            self.main_window._timer_overlay.set_pause_text(True)
        self.main_window.timer_status.setText("状态: 已重置并暂停，点击继续后开始计时")
        if self.main_window._timer_qtimer is not None:
            self.main_window._timer_qtimer.start(20)
