import json
import os
import sys

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QCheckBox, QComboBox, QSpinBox, QTextEdit, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QColor, QImage, QPalette, QPainter, QPixmap

import action
import cv2
import numpy as np


def _prepare_bg_pixmap(pixmap: QPixmap, opacity: float = 0.35) -> QPixmap:
    """对背景图降低不透明度，产生透明感，保留透明通道。"""
    if pixmap.isNull():
        return pixmap
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4)).copy()

    # 按 opacity 缩放 alpha 通道，0 完全透明，1 完全不透明
    arr[:, :, 3] = (arr[:, :, 3] * opacity).astype(np.uint8)

    q_image = QImage(arr.data, width, height, width * 4, QImage.Format.Format_ARGB32)
    return QPixmap.fromImage(q_image.copy())


class ContractLogContainer(QWidget):
    """在 paintEvent 中绘制背景图（带透明通道）的日志框容器。"""

    def __init__(self, bg_path: str, parent=None):
        super().__init__(parent)
        self._bg_pixmap = _prepare_bg_pixmap(QPixmap(bg_path))
        self._bg_color = QColor("#1e1e1e")
        self._contract_color = QColor("#fff0f0")
        self._use_contract = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet(
            "background-color: transparent; color: #d4d4d4; font-family: Consolas, monospace;"
        )
        layout.addWidget(self.text_edit)

    def set_contract_mode(self, use_contract: bool):
        self._use_contract = use_contract
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        color = self._contract_color if self._use_contract else self._bg_color
        painter.fillRect(self.rect(), color)
        if self._use_contract and not self._bg_pixmap.isNull():
            target_w = int(self.width())
            target_h = int(self.height())
            scaled = self._bg_pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        painter.end()
        super().paintEvent(event)


class ExecTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 脚本选择
        script_layout = QHBoxLayout()
        script_layout.addWidget(QLabel("脚本文件"))
        self.main_window.exec_script_path = QLineEdit()
        self.main_window.exec_script_path.setPlaceholderText("选择脚本 JSON 文件...")
        script_layout.addWidget(self.main_window.exec_script_path)
        self.main_window.btn_browse = QPushButton("浏览")
        self.main_window.btn_browse.clicked.connect(self._browse_script)
        script_layout.addWidget(self.main_window.btn_browse)
        layout.addLayout(script_layout)

        # 参数勾选
        params_group = QGroupBox("运行参数")
        params_layout = QHBoxLayout(params_group)
        self.main_window.chk_loop = QCheckBox("无限凸图 (--loop)")
        self.main_window.chk_leak = QCheckBox("漏怪检测 (--leak)")
        self.main_window.chk_debug = QCheckBox("Debug (--debug)")
        self.main_window.chk_direct_start = QCheckBox("直接开始作战 (--direct-start)")
        self.main_window.chk_challenge_mode = QCheckBox("突袭模式")
        self.main_window.chk_loop.stateChanged.connect(self._on_loop_changed)
        self.main_window.chk_direct_start.stateChanged.connect(self._on_direct_start_changed)
        self.main_window.chk_challenge_mode.stateChanged.connect(self._on_challenge_mode_changed)
        params_layout.addWidget(self.main_window.chk_loop)
        params_layout.addWidget(self.main_window.chk_leak)
        params_layout.addWidget(self.main_window.chk_debug)
        params_layout.addWidget(self.main_window.chk_direct_start)
        params_layout.addWidget(self.main_window.chk_challenge_mode)
        params_layout.addStretch()
        layout.addWidget(params_group)

        # 助战参数
        support_group = QGroupBox("助战参数")
        support_layout = QHBoxLayout(support_group)
        self.main_window.chk_borrow_support = QCheckBox("借用干员")
        self.main_window.chk_borrow_support.stateChanged.connect(self._on_borrow_support_changed)
        support_layout.addWidget(self.main_window.chk_borrow_support)

        support_layout.addWidget(QLabel("好友位置"))
        self.main_window.spin_support_friend = QSpinBox()
        self.main_window.spin_support_friend.setRange(0, 8)
        self.main_window.spin_support_friend.setEnabled(False)
        support_layout.addWidget(self.main_window.spin_support_friend)

        support_layout.addWidget(QLabel("携带技能"))
        self.main_window.combo_support_skill = QComboBox()
        self.main_window.combo_support_skill.addItems(["1", "2", "3"])
        self.main_window.combo_support_skill.setEnabled(False)
        support_layout.addWidget(self.main_window.combo_support_skill)

        support_layout.addWidget(QLabel("模组选择"))
        self.main_window.combo_support_module = QComboBox()
        self.main_window.combo_support_module.addItems(["1", "2", "3"])
        self.main_window.combo_support_module.setEnabled(False)
        support_layout.addWidget(self.main_window.combo_support_module)

        support_layout.addStretch()
        layout.addWidget(support_group)

        # 键位设置
        keys_group = QGroupBox("键位设置")
        keys_layout = QHBoxLayout(keys_group)

        keys_layout.addWidget(QLabel("暂停键"))
        self.main_window.combo_pause_key = QComboBox()
        self.main_window.combo_pause_key.setEditable(True)
        self.main_window.combo_pause_key.addItems(["p", "space", "q", "e", "r", "f"])
        self.main_window.combo_pause_key.setFixedWidth(80)
        keys_layout.addWidget(self.main_window.combo_pause_key)

        keys_layout.addWidget(QLabel("技能键"))
        self.main_window.line_skill_key = QLineEdit("e")
        self.main_window.line_skill_key.setMaxLength(8)
        self.main_window.line_skill_key.setFixedWidth(60)
        keys_layout.addWidget(self.main_window.line_skill_key)

        keys_layout.addWidget(QLabel("撤退键"))
        self.main_window.line_retreat_key = QLineEdit("q")
        self.main_window.line_retreat_key.setMaxLength(8)
        self.main_window.line_retreat_key.setFixedWidth(60)
        keys_layout.addWidget(self.main_window.line_retreat_key)

        self.main_window.combo_pause_key.currentTextChanged.connect(self._on_game_key_changed)
        self.main_window.line_skill_key.textChanged.connect(self._on_game_key_changed)
        self.main_window.line_retreat_key.textChanged.connect(self._on_game_key_changed)

        keys_layout.addStretch()
        layout.addWidget(keys_group)

        # 合约选项
        contract_group = QGroupBox("合约选项")
        contract_layout = QHBoxLayout(contract_group)
        contract_layout.addWidget(QLabel("费用回复 tag"))
        self.main_window.combo_cost_tag = QComboBox()
        self.main_window.combo_cost_tag.addItem("无", "")
        self.main_window.combo_cost_tag.addItem("费用回复降低25%", "cc_25")
        self.main_window.combo_cost_tag.addItem("费用回复降低50%", "cc_50")
        self.main_window.combo_cost_tag.addItem("费用回复降低75%", "cc_75")
        contract_layout.addWidget(self.main_window.combo_cost_tag)
        contract_layout.addStretch()
        layout.addWidget(contract_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.main_window.btn_run = QPushButton("运行脚本")
        self.main_window.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        self.main_window.btn_stop = QPushButton("停止")
        self.main_window.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.main_window.btn_run)
        btn_layout.addWidget(self.main_window.btn_stop)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 状态
        self.main_window.status_label = QLabel("状态: 就绪")
        layout.addWidget(self.main_window.status_label)

        # 日志
        contract_path = os.path.join(self.main_window._project_root(), "core", "resource", "Contract.png")
        self._log_container = ContractLogContainer(contract_path, self)
        self.main_window.log_text = self._log_container.text_edit
        layout.addWidget(self._log_container)

        # 背景图容器（暂时注释）
        # self.main_window.log_text = QTextEdit()
        # self.main_window.log_text.setReadOnly(True)
        # self.main_window.log_text.setStyleSheet(
        #     "background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;"
        # )
        # layout.addWidget(self.main_window.log_text)

        # 默认状态下勾选后使用绿色填充，未勾选时保持系统默认底色
        self._checkbox_default_checked_style = (
            "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #1a1a1a; border-radius: 3px; }"
            "QCheckBox::indicator:checked { background-color: #4CAF50; }"
        )
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(self._checkbox_default_checked_style)

        # 绑定
        self.main_window.btn_run.clicked.connect(self._start_script)
        self.main_window.btn_stop.clicked.connect(self._stop_script)
        self.main_window.combo_cost_tag.currentIndexChanged.connect(self._on_cost_tag_changed)

        self.main_window.process = None

    def _on_cost_tag_changed(self):
        """选择危机合约 tag 时通过 QPalette 切换暗红色主题，未选择时恢复默认。"""
        cost_tag = self.main_window.combo_cost_tag.currentData()
        if cost_tag:
            palette = QPalette(self.style().standardPalette())
            dark_red = QColor("#6b0000")
            dark_log_bg = QColor("#3b0000")
            light_red = QColor("#fff0f0")
            text_light = QColor("#f0f0f0")
            text_dark = QColor("#1a1a1a")
            palette.setColor(QPalette.ColorRole.Window, dark_red)
            palette.setColor(QPalette.ColorRole.Base, light_red)
            palette.setColor(QPalette.ColorRole.AlternateBase, dark_red)
            palette.setColor(QPalette.ColorRole.Text, text_dark)
            palette.setColor(QPalette.ColorRole.Button, dark_red)
            palette.setColor(QPalette.ColorRole.ButtonText, text_light)
            palette.setColor(QPalette.ColorRole.WindowText, text_light)
            self.setAutoFillBackground(True)
            self.setPalette(palette)
            for child in self.findChildren(QWidget):
                child.setPalette(palette)
            # 勾选框使用浅红底、白字
            checkbox_palette = QPalette(palette)
            checkbox_palette.setColor(QPalette.ColorRole.Window, light_red)
            checkbox_palette.setColor(QPalette.ColorRole.Button, light_red)
            checkbox_palette.setColor(QPalette.ColorRole.Base, light_red)
            checkbox_palette.setColor(QPalette.ColorRole.WindowText, text_light)
            checkbox_palette.setColor(QPalette.ColorRole.ButtonText, text_light)
            checkbox_palette.setColor(QPalette.ColorRole.Text, text_light)
            # 下拉框/数字框使用浅红底、深色字（与输入框保持一致）
            inputlike_palette = QPalette(palette)
            inputlike_palette.setColor(QPalette.ColorRole.Window, light_red)
            inputlike_palette.setColor(QPalette.ColorRole.Button, light_red)
            inputlike_palette.setColor(QPalette.ColorRole.Base, light_red)
            inputlike_palette.setColor(QPalette.ColorRole.WindowText, text_dark)
            inputlike_palette.setColor(QPalette.ColorRole.ButtonText, text_dark)
            inputlike_palette.setColor(QPalette.ColorRole.Text, text_dark)
            checkbox_indicator_style = (
                "QCheckBox { color: #f0f0f0; }"
                "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #1a1a1a; border-radius: 3px; background-color: #fff0f0; }"
                "QCheckBox::indicator:checked { background-color: #4CAF50; border: 1px solid #1a1a1a; border-radius: 3px; }"
            )
            for checkbox in self.findChildren(QCheckBox):
                checkbox.setStyleSheet(checkbox_indicator_style)
                checkbox.setPalette(checkbox_palette)
            for combobox in self.findChildren(QComboBox):
                combobox.setPalette(inputlike_palette)
            for spinbox in self.findChildren(QSpinBox):
                spinbox.setPalette(inputlike_palette)
            self.main_window.log_text.setStyleSheet(
                f"background-color: transparent; color: {text_dark.name()}; font-family: Consolas, monospace;"
            )
            self._log_container.set_contract_mode(True)
            # self.main_window.log_text.setStyleSheet(
            #     f"background-color: {dark_log_bg.name()}; color: {text_light.name()}; font-family: Consolas, monospace;"
            # )
        else:
            self.setAutoFillBackground(False)
            standard = self.style().standardPalette()
            self.setPalette(standard)
            for checkbox in self.findChildren(QCheckBox):
                checkbox.setStyleSheet(self._checkbox_default_checked_style)
                checkbox.style().unpolish(checkbox)
                checkbox.style().polish(checkbox)
                checkbox.update()
            for child in self.findChildren(QWidget):
                child.setPalette(QPalette())
            self.style().unpolish(self)
            self.style().polish(self)
            self.repaint()
            self.main_window.log_text.setStyleSheet(
                "background-color: transparent; color: #d4d4d4; font-family: Consolas, monospace;"
            )
            self._log_container.set_contract_mode(False)

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(self.main_window, "选择脚本", "", "JSON (*.json)")
        if path:
            self.main_window.exec_script_path.setText(path)

    def _on_borrow_support_changed(self, state):
        if isinstance(state, Qt.CheckState):
            enabled = state == Qt.CheckState.Checked
        else:
            enabled = state == Qt.CheckState.Checked.value
        if self.main_window.chk_direct_start.isChecked():
            enabled = False
        self.main_window.spin_support_friend.setEnabled(enabled)
        self.main_window.combo_support_skill.setEnabled(enabled)
        self.main_window.combo_support_module.setEnabled(enabled)

    def _on_loop_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.main_window.chk_direct_start.setChecked(False)
            self.main_window.chk_direct_start.setEnabled(False)
        else:
            self.main_window.chk_direct_start.setEnabled(True)

    def _on_direct_start_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.main_window.chk_loop.setChecked(False)
            self.main_window.chk_loop.setEnabled(False)
            self.main_window.chk_challenge_mode.setChecked(False)
            self.main_window.chk_challenge_mode.setEnabled(False)
        else:
            self.main_window.chk_loop.setEnabled(True)
            self.main_window.chk_challenge_mode.setEnabled(True)
        self.main_window.chk_borrow_support.setEnabled(not checked)
        self._on_borrow_support_changed(self.main_window.chk_borrow_support.checkState())

    def _on_challenge_mode_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.main_window.chk_direct_start.setChecked(False)
            self.main_window.chk_direct_start.setEnabled(False)
        else:
            self.main_window.chk_direct_start.setEnabled(True)

    def _on_game_key_changed(self, text):
        game_keys = self.main_window._game_key_set()
        widgets = [
            (self.main_window.line_matchstick_select, self.main_window.chk_matchstick_select, "选中干员"),
            (self.main_window.line_matchstick_166, self.main_window.chk_matchstick_166, "过 166ms"),
            (self.main_window.line_matchstick_50, self.main_window.chk_matchstick_50, "过 50ms"),
        ]
        for line, chk, name in widgets:
            hotkey = self.main_window._normalize_hotkey(line.text())
            if hotkey and "+" not in hotkey and hotkey in game_keys and chk.isChecked():
                QMessageBox.warning(
                    self.main_window,
                    "热键冲突",
                    f"游戏内快捷键与划火柴热键 '{name} ({hotkey})' 冲突，已自动禁用该划火柴热键。",
                )
                chk.setChecked(False)
        self.main_window._apply_matchstick_config()
        action.configure_keys(
            pause=self.main_window._normalize_key_name(self.main_window.combo_pause_key.currentText()),
            skill=self.main_window._normalize_key_name(self.main_window.line_skill_key.text()),
            retreat=self.main_window._normalize_key_name(self.main_window.line_retreat_key.text()),
        )

    def _on_matchstick_enabled_changed(self, state):
        self.main_window._apply_matchstick_config()

    def _on_matchstick_hotkey_changed(self, text):
        sender = self.sender()
        op_map = {
            self.main_window.line_matchstick_select: ("选中干员", self.main_window.chk_matchstick_select),
            self.main_window.line_matchstick_166: ("过 166ms", self.main_window.chk_matchstick_166),
            self.main_window.line_matchstick_50: ("过 50ms", self.main_window.chk_matchstick_50),
        }
        op_name, chk = op_map.get(sender, ("", None))
        normalized = self.main_window._normalize_hotkey(text)
        game_keys = self.main_window._game_key_set()
        if normalized and "+" not in normalized and normalized in game_keys:
            QMessageBox.warning(self.main_window, "热键冲突", f"'{op_name}' 热键 '{normalized}' 与脚本键位设置中的游戏快捷键冲突，请更换。")
            sender.blockSignals(True)
            cfg = action.get_matchstick_config()
            sender.setText(cfg["hotkeys"].get(self._matchstick_op_from_widget(sender), ""))
            sender.blockSignals(False)
            return
        self.main_window._apply_matchstick_config()

    def _matchstick_op_from_widget(self, widget):
        if widget is self.main_window.line_matchstick_select:
            return "select_operator"
        if widget is self.main_window.line_matchstick_166:
            return "pass_166ms"
        if widget is self.main_window.line_matchstick_50:
            return "pass_50ms"
        return ""

    def _start_script(self):
        script_path = self.main_window.exec_script_path.text()
        if not script_path:
            QMessageBox.warning(self.main_window, "警告", "请先选择脚本文件")
            return

        self.main_window._save_config()

        args = ["--run-script", script_path]
        if self.main_window.chk_loop.isChecked():
            args.append("--loop")
        if self.main_window.chk_leak.isChecked():
            args.append("--leak")
        if self.main_window.chk_debug.isChecked():
            args.append("--debug")
        if self.main_window.chk_direct_start.isChecked():
            args.append("--direct-start")
        if self.main_window.chk_challenge_mode.isChecked():
            args.append("--challenge-mode")
        cost_tag = self.main_window.combo_cost_tag.currentData()
        if cost_tag:
            args.extend(["--cost-tag", cost_tag])
        if self.main_window.chk_borrow_support.isChecked():
            args.append("--borrow-support")
            args.extend([
                "--support-friend-index",
                str(self.main_window.spin_support_friend.value()),
                "--support-skill",
                self.main_window.combo_support_skill.currentText(),
                "--support-module",
                self.main_window.combo_support_module.currentText(),
            ])

        args.extend([
            "--pause-key", self.main_window._normalize_key_name(self.main_window.combo_pause_key.currentText()),
            "--skill-key", self.main_window._normalize_key_name(self.main_window.line_skill_key.text()),
            "--retreat-key", self.main_window._normalize_key_name(self.main_window.line_retreat_key.text()),
        ])

        self.main_window.process = QProcess()
        self.main_window.process.setWorkingDirectory(self.main_window._project_root())
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.main_window.process.setProcessEnvironment(env)
        self.main_window.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.main_window.process.readyReadStandardOutput.connect(self._on_stdout)
        self.main_window.process.finished.connect(self._on_finished)

        self.main_window.log_text.clear()
        if self.main_window.chk_direct_start.isChecked():
            self.main_window.log_text.append("[系统] 直接开始作战模式，脚本初始化中...")
        else:
            self.main_window.log_text.append("[系统] 脚本初始化中，首次加载 OCR 模型可能需要十几秒...")
        if getattr(sys, "frozen", False):
            self.main_window.process.start(sys.executable, args)
        else:
            self.main_window.process.start(sys.executable, ["entry.py"] + args)

        self.main_window.status_label.setText("状态: 脚本初始化中...")
        self.main_window.btn_run.setEnabled(False)
        self.main_window.btn_stop.setEnabled(True)
        self.main_window._has_minimized = False

    def _on_stdout(self):
        data = self.main_window.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        lines = data.rstrip().splitlines()
        filtered = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if "[系统] 脚本开始运行" in stripped:
                self.main_window.status_label.setText("状态: 运行中")
            if "[OCR] 初始化成功" in stripped and not self.main_window._has_minimized:
                self.main_window._has_minimized = True
                self.main_window.showMinimized()
            if any(p in stripped for p in ("Creating model", "Model files already exist", "To redownload", "Loading weights", "%|", "[32m", "[0m", "[OCR] 初始化成功")):
                continue
            if stripped.startswith("__TIMER_SHIELD__:"):
                try:
                    duration_ms = float(stripped.split(":", 1)[1])
                    if self.main_window._region_timer is not None and self.main_window._region_timer.is_running():
                        self.main_window._region_timer.shield_matchstick(duration_ms)
                except Exception:
                    pass
                continue
            if stripped.startswith("__TIMER_ADJUST__:"):
                try:
                    offset_ms = float(stripped.split(":", 1)[1])
                    if self.main_window._region_timer is not None and self.main_window._region_timer.is_running():
                        self.main_window._region_timer.adjust(offset_ms)
                except Exception:
                    pass
                continue
            filtered.append(stripped)
        if filtered:
            self.main_window.log_text.append("\n".join(filtered))

    def _on_stderr(self):
        data = self.main_window.process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data.strip():
            self.main_window.log_text.append(f"[stderr] {data.strip()}")

    def _on_finished(self, exit_code, exit_status):
        self.main_window.status_label.setText(f"状态: 已停止 (退出码 {exit_code})")
        self.main_window.btn_run.setEnabled(True)
        self.main_window.btn_stop.setEnabled(False)
        action.start_matchstick_listener()
        if self.main_window._region_timer is not None and self.main_window._region_timer.is_running():
            self.main_window._region_timer.reconnect_hotkey()

    def _stop_script(self):
        if self.main_window.process and self.main_window.process.state() != QProcess.ProcessState.NotRunning:
            self.main_window.process.terminate()
            QTimer.singleShot(3000, self._force_kill)

    def _force_kill(self):
        if self.main_window.process and self.main_window.process.state() != QProcess.ProcessState.NotRunning:
            self.main_window.process.kill()
            self.main_window.log_text.append("[系统] 强制终止进程")
