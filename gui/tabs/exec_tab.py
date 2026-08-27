import json
import os
import sys

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit, QPushButton,
    QGroupBox, QCheckBox, QComboBox, QSpinBox, QTextEdit, QMessageBox, QFileDialog,
    QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QColor, QFont, QImage, QPalette, QPainter, QPixmap

import action
import cv2
import numpy as np

from core.base.paths import gui_template
from gui.tabs._ui_utils import scaled_icon_path, IconGroupBox
from models.script_schema import ScriptModel


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


class SupportConfigDialog(QDialog):
    """脚本执行页助战参数配置对话框。"""

    def __init__(self, parent=None, friend_index=0, skill=1, module=1):
        super().__init__(parent)
        self.setWindowTitle("助战配置")
        self.setMinimumWidth(280)
        self._friend_index = friend_index
        self._skill = skill
        self._module = module
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("好友位置"), 0, 0)
        self.spin_friend = QSpinBox()
        self.spin_friend.setRange(0, 8)
        self.spin_friend.setValue(self._friend_index)
        form_layout.addWidget(self.spin_friend, 0, 1)

        form_layout.addWidget(QLabel("携带技能"), 1, 0)
        self.combo_skill = QComboBox()
        self.combo_skill.addItems(["1", "2", "3"])
        self.combo_skill.setCurrentIndex(max(0, self._skill - 1))
        form_layout.addWidget(self.combo_skill, 1, 1)

        form_layout.addWidget(QLabel("模组选择"), 2, 0)
        self.combo_module = QComboBox()
        self.combo_module.addItems(["1", "2", "3"])
        self.combo_module.setCurrentIndex(max(0, self._module - 1))
        form_layout.addWidget(self.combo_module, 2, 1)
        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_config(self) -> dict:
        return {
            "friend_index": self.spin_friend.value(),
            "skill": int(self.combo_skill.currentText()),
            "module": int(self.combo_module.currentText()),
        }


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
        self._init_error_shown = False
        self._user_stopped = False
        self._last_lines: list[str] = []
        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(12)

        # 左侧面板：配置区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # 脚本选择
        script_group = IconGroupBox("脚本选择", "choose.png")
        script_layout = QHBoxLayout(script_group)
        self.main_window.exec_script_path = QLineEdit()
        self.main_window.exec_script_path.setPlaceholderText("选择脚本 JSON 文件...")
        script_layout.addWidget(self.main_window.exec_script_path)
        self.main_window.btn_browse = QPushButton("浏览")
        self.main_window.btn_browse.clicked.connect(self._browse_script)
        script_layout.addWidget(self.main_window.btn_browse)
        left_layout.addWidget(script_group)

        # 参数勾选
        params_group = IconGroupBox("运行参数", "level_setup.png")
        params_layout = QGridLayout(params_group)
        self.main_window.chk_loop = QCheckBox("无限凸图")
        self.main_window.chk_leak = QCheckBox("漏怪检测")
        self.main_window.chk_debug = QCheckBox("Debug")
        self.main_window.chk_direct_start = QCheckBox("直接开始作战")
        self.main_window.chk_challenge_mode = QCheckBox("突袭模式")
        self.main_window.chk_sand_table = QCheckBox("沙盘推演")
        self.main_window.chk_speed2x = QCheckBox("二倍速凸图")
        self.main_window.chk_loop.stateChanged.connect(self._on_loop_changed)
        self.main_window.chk_direct_start.stateChanged.connect(self._on_direct_start_changed)
        self.main_window.chk_challenge_mode.stateChanged.connect(self._on_challenge_mode_changed)
        self.main_window.chk_sand_table.stateChanged.connect(self._on_sand_table_changed)
        params_layout.addWidget(self.main_window.chk_loop, 0, 0)
        params_layout.addWidget(self.main_window.chk_leak, 0, 1)
        params_layout.addWidget(self.main_window.chk_direct_start, 0, 2)
        params_layout.addWidget(self.main_window.chk_challenge_mode, 1, 0)
        params_layout.addWidget(self.main_window.chk_sand_table, 1, 1)
        params_layout.addWidget(self.main_window.chk_speed2x, 1, 2)
        left_layout.addWidget(params_group)

        # Debug
        debug_group = IconGroupBox("Debug", "debug.png")
        debug_layout = QHBoxLayout(debug_group)
        self.main_window.chk_debug.stateChanged.connect(self.main_window._save_config)
        debug_layout.addWidget(QLabel("Debug输出"))
        debug_layout.addWidget(self.main_window.chk_debug)
        debug_layout.addStretch()
        left_layout.addWidget(debug_group)

        # 助战参数
        support_group = IconGroupBox("助战参数", "support_operator.png")
        support_layout = QGridLayout(support_group)
        self.main_window.chk_borrow_support = QCheckBox("借用干员")
        self.main_window.chk_borrow_support.stateChanged.connect(self._on_borrow_support_changed)
        support_layout.addWidget(QLabel("借用干员"), 0, 0)
        support_layout.addWidget(self.main_window.chk_borrow_support, 0, 1)

        self.main_window.btn_support_config = QPushButton("配置")
        self.main_window.btn_support_config.setEnabled(False)
        self.main_window.btn_support_config.setFixedWidth(60)
        self.main_window.btn_support_config.clicked.connect(self._on_support_config_clicked)
        support_layout.addWidget(QLabel("助战配置"), 0, 2)
        support_layout.addWidget(self.main_window.btn_support_config, 0, 3)
        support_layout.setColumnStretch(4, 1)

        # 助战配置值保存在隐藏的输入控件中，便于主窗口统一读写配置
        self.main_window.spin_support_friend = QSpinBox()
        self.main_window.spin_support_friend.setRange(0, 8)
        self.main_window.spin_support_friend.setVisible(False)

        self.main_window.combo_support_skill = QComboBox()
        self.main_window.combo_support_skill.addItems(["1", "2", "3"])
        self.main_window.combo_support_skill.setVisible(False)

        self.main_window.combo_support_module = QComboBox()
        self.main_window.combo_support_module.addItems(["1", "2", "3"])
        self.main_window.combo_support_module.setVisible(False)

        left_layout.addWidget(support_group)

        # 键位设置
        keys_group = IconGroupBox("键位设置", "keyboard.png")
        keys_layout = QGridLayout(keys_group)
        self.main_window.combo_pause_key = QComboBox()
        self.main_window.combo_pause_key.setEditable(True)
        self.main_window.combo_pause_key.addItems(["space", "p", "q", "e", "r", "f"])
        self.main_window.combo_pause_key.setFixedWidth(80)
        keys_layout.addWidget(QLabel("暂停键"), 0, 0)
        keys_layout.addWidget(self.main_window.combo_pause_key, 0, 1)

        self.main_window.line_skill_key = QLineEdit("e")
        self.main_window.line_skill_key.setMaxLength(8)
        self.main_window.line_skill_key.setFixedWidth(60)
        keys_layout.addWidget(QLabel("技能键"), 0, 2)
        keys_layout.addWidget(self.main_window.line_skill_key, 0, 3)

        self.main_window.line_retreat_key = QLineEdit("q")
        self.main_window.line_retreat_key.setMaxLength(8)
        self.main_window.line_retreat_key.setFixedWidth(60)
        keys_layout.addWidget(QLabel("撤退键"), 1, 0)
        keys_layout.addWidget(self.main_window.line_retreat_key, 1, 1)

        self.main_window.line_speed_key = QLineEdit("f")
        self.main_window.line_speed_key.setMaxLength(8)
        self.main_window.line_speed_key.setFixedWidth(60)
        keys_layout.addWidget(QLabel("倍速键"), 1, 2)
        keys_layout.addWidget(self.main_window.line_speed_key, 1, 3)

        self.main_window.combo_pause_key.currentTextChanged.connect(self._on_game_key_changed)
        self.main_window.line_skill_key.textChanged.connect(self._on_game_key_changed)
        self.main_window.line_retreat_key.textChanged.connect(self._on_game_key_changed)
        self.main_window.line_speed_key.textChanged.connect(self._on_game_key_changed)
        left_layout.addWidget(keys_group)

        # 合约选项
        contract_group = IconGroupBox("合约选项", "warning.png")
        contract_layout = QHBoxLayout(contract_group)
        contract_layout.addWidget(QLabel("费用回复 tag"))
        self.main_window.combo_cost_tag = QComboBox()
        self.main_window.combo_cost_tag.addItem("无", "")
        self.main_window.combo_cost_tag.addItem("费用回复降低25%", "cc_25")
        self.main_window.combo_cost_tag.addItem("费用回复降低50%", "cc_50")
        self.main_window.combo_cost_tag.addItem("费用回复降低75%", "cc_75")
        self.main_window.combo_cost_tag.addItem("费用不自然回复", "no_regen")
        contract_layout.addWidget(self.main_window.combo_cost_tag)
        contract_layout.addStretch()
        left_layout.addWidget(contract_group)

        # 按钮 + 状态
        run_group = IconGroupBox("执行控制", "battle.png")
        run_layout = QVBoxLayout(run_group)
        btn_layout = QHBoxLayout()
        self.main_window.btn_run = QPushButton("运行脚本")
        self.main_window.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        self.main_window.btn_stop = QPushButton("停止")
        self.main_window.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.main_window.btn_run)
        btn_layout.addWidget(self.main_window.btn_stop)
        btn_layout.addStretch()
        run_layout.addLayout(btn_layout)

        self.main_window.status_label = QLabel("状态: 就绪")
        run_layout.addWidget(self.main_window.status_label)
        left_layout.addWidget(run_group)

        left_layout.addStretch()
        main_layout.addWidget(left_widget, 1)

        # 右侧面板：日志区
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        log_title_layout = QHBoxLayout()
        log_title_layout.setSpacing(4)
        log_icon = QLabel()
        log_icon.setPixmap(
            QPixmap(scaled_icon_path("log.png", 20)).scaled(
                20, 20,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        log_title_layout.addWidget(log_icon)
        log_title_label = QLabel("执行日志")
        log_title_label.setStyleSheet("font-weight: bold;")
        log_title_layout.addWidget(log_title_label)
        log_title_layout.addStretch()
        right_layout.addLayout(log_title_layout)
        contract_path = str(gui_template("Contract.png"))
        self._log_container = ContractLogContainer(contract_path, self)
        self.main_window.log_text = self._log_container.text_edit
        right_layout.addWidget(self._log_container)
        main_layout.addLayout(right_layout, 2)

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
        """费用回复 tag 切换（保留接口，当前不应用主题变色）。"""
        pass

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
        self.main_window.btn_support_config.setEnabled(enabled)
        self.main_window.spin_support_friend.setEnabled(enabled)
        self.main_window.combo_support_skill.setEnabled(enabled)
        self.main_window.combo_support_module.setEnabled(enabled)

    def _on_support_config_clicked(self):
        dialog = SupportConfigDialog(
            self.main_window,
            friend_index=self.main_window.spin_support_friend.value(),
            skill=int(self.main_window.combo_support_skill.currentText()),
            module=int(self.main_window.combo_support_module.currentText()),
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            cfg = dialog.get_config()
            self.main_window.spin_support_friend.setValue(cfg["friend_index"])
            self.main_window.combo_support_skill.setCurrentIndex(cfg["skill"] - 1)
            self.main_window.combo_support_module.setCurrentIndex(cfg["module"] - 1)
            self.main_window._save_config()

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
            self.main_window.chk_sand_table.setChecked(False)
            self.main_window.chk_sand_table.setEnabled(False)
        else:
            self.main_window.chk_loop.setEnabled(True)
            self.main_window.chk_challenge_mode.setEnabled(True)
            self.main_window.chk_sand_table.setEnabled(True)
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
            self.main_window.chk_sand_table.setChecked(False)
            self.main_window.chk_sand_table.setEnabled(False)
        else:
            self.main_window.chk_direct_start.setEnabled(True)
            if not self.main_window.chk_sand_table.isChecked():
                self.main_window.chk_sand_table.setEnabled(True)

    def _on_sand_table_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.main_window.chk_direct_start.setChecked(False)
            self.main_window.chk_direct_start.setEnabled(False)
            self.main_window.chk_challenge_mode.setChecked(False)
            self.main_window.chk_challenge_mode.setEnabled(False)
        else:
            self.main_window.chk_direct_start.setEnabled(True)
            if not self.main_window.chk_challenge_mode.isChecked():
                self.main_window.chk_challenge_mode.setEnabled(True)

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
            speed=self.main_window._normalize_key_name(self.main_window.line_speed_key.text()),
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

        try:
            with open(script_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            script = ScriptModel.model_validate(data)
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "脚本加载失败",
                f"无法读取脚本 {script_path}:\n{e}",
            )
            return

        dir_issues = script.validate_deploy_directions()
        if dir_issues:
            dir_lines = []
            for time_ms, category, name in dir_issues:
                s, f = self.main_window._ms_to_sf(time_ms)
                dir_lines.append(f"{s}秒第{f}帧：{category} {name} 部署缺少方向")
            msg = QMessageBox(self.main_window)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setWindowTitle("脚本校验警告")
            msg.setText(
                "以下部署动作缺少方向参数（干员必须填写；道具/召唤物因半数以上部署有方向，视为需要方向）：\n"
                + "\n".join(dir_lines)
            )
            run_btn = msg.addButton("继续执行", QMessageBox.ButtonRole.AcceptRole)
            cancel_btn = msg.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            msg.setDefaultButton(cancel_btn)
            msg.exec()
            if msg.clickedButton() != run_btn:
                return

        self.main_window._save_config()
        self._init_error_shown = False
        self._user_stopped = False
        self._last_lines.clear()

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
        if self.main_window.chk_sand_table.isChecked():
            args.append("--sand-table")
        if self.main_window.chk_speed2x.isChecked():
            args.append("--speed2x")
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
            "--speed-key", self.main_window._normalize_key_name(self.main_window.line_speed_key.text()),
        ])

        self.main_window.process = QProcess()
        self.main_window.process.setWorkingDirectory(self.main_window._project_root())
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        self.main_window.process.setProcessEnvironment(env)
        self.main_window.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.main_window.process.readyReadStandardOutput.connect(self._on_stdout)
        self.main_window.process.finished.connect(self._on_finished)
        self.main_window.process.errorOccurred.connect(self._on_process_error)

        self.main_window.log_text.clear()
        cmd_preview = " ".join([sys.executable] + (["entry.py"] if not getattr(sys, "frozen", False) else []) + args)
        self.main_window.log_text.append(f"[系统] 启动参数: {cmd_preview}")
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
            self._last_lines.append(stripped)
            if len(self._last_lines) > 200:
                self._last_lines.pop(0)
            if stripped.startswith("__INIT_ERROR__:"):
                error_msg = stripped.split(":", 1)[1].strip()
                self._show_init_error(error_msg)
                continue
            if "[系统] 脚本开始运行" in stripped:
                self.main_window.status_label.setText("状态: 运行中")
            if "[费用条同步]" in stripped and not self.main_window._has_minimized:
                self.main_window._has_minimized = True
                QTimer.singleShot(0, self.main_window.showMinimized)
            if any(p in stripped for p in ("Creating model", "Model files already exist", "To redownload", "Loading weights", "%|", "[32m", "[0m")):
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

    def _show_init_error(self, error_msg: str):
        """子进程初始化失败时弹窗提示，避免打包后无控制台看不到报错。"""
        if self._init_error_shown:
            return
        self._init_error_shown = True
        detail = "\n".join(self._last_lines[-30:])
        QMessageBox.critical(
            self.main_window,
            "脚本初始化失败",
            f"后端进程初始化时出错，模型或资源可能未正确加载：\n\n{error_msg}\n\n"
            f"最近日志（已同步到下方输出框）：\n{detail}",
        )

    def _on_stderr(self):
        data = self.main_window.process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data.strip():
            self.main_window.log_text.append(f"[stderr] {data.strip()}")

    def _on_process_error(self, error):
        """QProcess 自身启动失败时弹窗提示（如 exe 缺失、权限不足等）。

        用户手动停止时强制 kill 也可能触发 Crashed，此时跳过弹窗。
        """
        if self._user_stopped:
            return
        error_text = self.main_window.process.errorString()
        QMessageBox.critical(
            self.main_window,
            "无法启动脚本进程",
            f"启动后端进程失败：\n{error_text}\n\n"
            f"错误码：{error}\n\n"
            f"请检查打包文件是否完整，或尝试重新打包。",
        )
        self.main_window.status_label.setText(f"状态: 启动失败 ({error_text})")
        self.main_window.btn_run.setEnabled(True)
        self.main_window.btn_stop.setEnabled(False)

    def _on_finished(self, exit_code, exit_status):
        self.main_window.status_label.setText(f"状态: 已停止 (退出码 {exit_code})")
        self.main_window.btn_run.setEnabled(True)
        self.main_window.btn_stop.setEnabled(False)
        action.start_matchstick_listener()
        if self.main_window._region_timer is not None and self.main_window._region_timer.is_running():
            self.main_window._region_timer.reconnect_hotkey()

        if exit_code != 0 and not self._init_error_shown and not self._user_stopped:
            detail = "\n".join(self._last_lines[-30:])
            QMessageBox.critical(
                self.main_window,
                "脚本异常退出",
                f"后端进程异常退出（退出码 {exit_code}）。\n\n"
                f"最近日志：\n{detail}\n\n"
                f"请检查下方输出框中的完整报错信息。",
            )
        self._init_error_shown = False
        self._last_lines.clear()

    def _stop_script(self):
        self._user_stopped = True
        if self.main_window.process and self.main_window.process.state() != QProcess.ProcessState.NotRunning:
            self.main_window.process.terminate()
            QTimer.singleShot(3000, self._force_kill)

    def _force_kill(self):
        if self.main_window.process and self.main_window.process.state() != QProcess.ProcessState.NotRunning:
            self.main_window.process.kill()
            self.main_window.log_text.append("[系统] 强制终止进程")
