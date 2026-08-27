import action
from core.capture.capture import WindowCapture
from core.game_state.region_state_timer import RegionStateTimer
from gui._window_effects import (
    remove_dwm_glass_border,
    set_window_topmost,
    set_tool_window_style,
)
from gui.tabs._ui_utils import IconGroupBox
from gui.timer_overlay import TimerOverlay

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QCheckBox, QComboBox,
    QGroupBox, QSizePolicy, QTextEdit, QMessageBox,
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
                "倍速区域 : 控制倍率 2.0x / 1.0x / 0.2x"
            )
        )

        self.main_window.chk_timer_high_perf = QCheckBox("高精度模式")
        self.main_window.chk_timer_high_perf.setToolTip(
            "启用独立 TimeKeeper 线程和 1ms 系统定时器分辨率，"
            "降低 MAX 费用条后的计时漂移（CPU 占用略高）。\n"
            "高精度模式下若需部署干员，建议先暂停游戏再部署，"
            "避免拖拽操作引入计时偏差。"
        )
        layout.addWidget(self.main_window.chk_timer_high_perf)

        contract_group = IconGroupBox("计时器费用条 tag", "warning.png")
        contract_layout = QHBoxLayout(contract_group)
        contract_layout.setContentsMargins(6, 6, 6, 6)
        self.main_window.combo_timer_cost_tag = QComboBox()
        self.main_window.combo_timer_cost_tag.addItem("无", "")
        self.main_window.combo_timer_cost_tag.addItem("费用回复降低25%", "cc_25")
        self.main_window.combo_timer_cost_tag.addItem("费用回复降低50%", "cc_50")
        self.main_window.combo_timer_cost_tag.addItem("费用回复降低75%", "cc_75")
        self.main_window.combo_timer_cost_tag.addItem("费用不自然回复", "no_regen")
        contract_layout.addWidget(QLabel("费用回复 tag"))
        contract_layout.addWidget(self.main_window.combo_timer_cost_tag)
        contract_layout.addStretch()
        layout.addWidget(contract_group)

        debug_group = IconGroupBox("Debug", "debug.png")
        debug_layout = QHBoxLayout(debug_group)
        debug_layout.setContentsMargins(6, 6, 6, 6)
        self.main_window.chk_timer_debug = QCheckBox("Debug")
        debug_layout.addWidget(QLabel("Debug输出"))
        debug_layout.addWidget(self.main_window.chk_timer_debug)
        debug_layout.addStretch()
        layout.addWidget(debug_group)

        control_group = IconGroupBox("计时控制", "battle.png")
        control_layout = QVBoxLayout(control_group)
        control_layout.setContentsMargins(6, 6, 6, 6)
        control_layout.setSpacing(6)

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
        control_layout.addLayout(btn_layout)

        self.main_window.timer_status = QLabel("状态: 就绪")
        control_layout.addWidget(self.main_window.timer_status)
        layout.addWidget(control_group)

        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml(
            """
            <h3>使用说明</h3>
            <ul>
                <li>在编队界面，点击<b>开始计时</b>后，主窗口会自动最小化，并在屏幕左上角显示悬浮计时器。</li>
                <li>计时器窗口默认置顶，可拖动到任意位置，不会被游戏遮挡。</li>
                <li>检测到正式开始游戏(费用条开始动)后自动开始计时。</li>
                <li>对三个区域持续模板监控，<span style="color: red;">因此请使用快捷键控制倍速和暂停，不要鼠标操作暂停键和倍率区，不要鼠标触碰到费用条。</span></li>
                <li>高精度模式下若需部署干员，<span style="color: red;">请先暂停游戏再部署</span>，避免拖拽导致的计时偏差。</li>
                <li>在费用条满之前会持续对费用条监控，通常不会存在误差；费用条满后不再监控费用条，通常情况下误差会在1帧内，刻意反复频繁操作会增大误差。</li>
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
            "select_operator": 3.0,
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
        if self.main_window._region_timer is not None:
            if self.main_window._region_timer.is_running():
                # 已经在运行，只需确保悬浮窗显示并置顶
                if self.main_window._timer_overlay is not None:
                    self.main_window._timer_overlay.show()
                    set_window_topmost(self.main_window._timer_overlay)
                return
            # 残留未运行的计时器，先清理避免状态混乱
            self.main_window._region_timer.stop()
            self.main_window._region_timer = None

        try:
            self.main_window._timer_capture = WindowCapture(
                backend="mss",
                debug=self.main_window.chk_timer_debug.isChecked(),
            )
            matchstick_hotkeys = self._build_matchstick_hotkeys()

            cost_tag = self.main_window.combo_timer_cost_tag.currentData() or None

            self.main_window._region_timer = RegionStateTimer(
                self.main_window._timer_capture,
                pause_key=action.pause_key(),
                debug=self.main_window.chk_timer_debug.isChecked(),
                matchstick_hotkeys=matchstick_hotkeys if matchstick_hotkeys else None,
                cost_bar_calibration_name=cost_tag,
                high_precision=self.main_window.chk_timer_high_perf.isChecked(),
            )
            self.main_window._region_timer.start(use_cost_detection=True)
        except Exception as e:
            QMessageBox.critical(self.main_window, "启动失败", f"计时器初始化失败:\n{e}")
            return

        self.main_window._timer_overlay = TimerOverlay(
            on_pause_clicked=self._toggle_timer_pause,
            on_reset_clicked=self._reset_region_timer,
            on_stop_clicked=self._stop_region_timer,
            debug=self.main_window.chk_timer_debug.isChecked(),
        )
        self.main_window._timer_overlay.show()
        remove_dwm_glass_border(self.main_window._timer_overlay)
        set_tool_window_style(self.main_window._timer_overlay)
        set_window_topmost(self.main_window._timer_overlay)
        # 延迟最小化主窗口，避免刚创建的悬浮窗受主窗口状态切换影响而无法立即显示
        QTimer.singleShot(100, self.main_window.showMinimized)

        self.main_window._timer_qtimer = QTimer(self.main_window)
        self.main_window._timer_qtimer.setTimerType(Qt.TimerType.PreciseTimer)
        self.main_window._timer_qtimer.timeout.connect(self._on_timer_tick)
        # 界面刷新改为 33ms，与游戏 30fps 对齐；区域 B 采样仍由 RegionStateTimer 独立线程负责
        self.main_window._timer_qtimer.start(33)

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
        # 关闭悬浮窗后强制主窗口回到前台并激活焦点，避免输入框无法点击
        self.main_window.raise_()
        self.main_window.activateWindow()

    def _reset_region_timer(self):
        if self.main_window._region_timer is None:
            return
        if self.main_window._timer_qtimer is not None:
            self.main_window._timer_qtimer.stop()
        # 先彻底停止再重新启动，避免旧 TimeKeeper/采样线程残留导致状态混乱
        self.main_window._region_timer.stop()
        self.main_window._region_timer.start(use_cost_detection=True)
        self.main_window._region_timer.manual_pause()
        if self.main_window._timer_overlay is not None:
            self.main_window._timer_overlay.update_time(0.0, 0, 0, 1.0, True)
            self.main_window._timer_overlay.set_pause_text(True)
        self.main_window.timer_status.setText("状态: 已重置并暂停，点击继续后开始计时")
        if self.main_window._timer_qtimer is not None:
            self.main_window._timer_qtimer.start(33)
