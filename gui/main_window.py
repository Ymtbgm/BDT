import json
import os
import shutil
import sys
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox,
    QFileDialog, QSpinBox, QComboBox, QTabWidget,
    QCheckBox, QTextEdit, QGroupBox, QListWidget,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QIcon

import action
from core.capture import WindowCapture
from core.region_state_timer import RegionStateTimer
from core.recorder import ActionRecorder
from gui.timer_overlay import TimerOverlay
from models.script_schema import ScriptModel, OperatorAction, ActionType, ItemInfo


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arknights Auto")
        # 兼容开发环境与 PyInstaller onedir 打包后的图标路径
        _icon_candidates = []
        _dev_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _icon_candidates.append(os.path.join(_dev_root, "core", "resource", "Icon.ico"))
        _meipass = getattr(sys, '_MEIPASS', None)
        if _meipass:
            _icon_candidates.append(os.path.join(os.path.dirname(_meipass), "core", "resource", "Icon.ico"))
            _icon_candidates.append(os.path.join(_meipass, "core", "resource", "Icon.ico"))
        _icon_candidates.append(os.path.join(os.getcwd(), "core", "resource", "Icon.ico"))
        _icon_path = next((p for p in _icon_candidates if os.path.exists(p)), _icon_candidates[0])
        self.setWindowIcon(QIcon(_icon_path))
        self.resize(1200, 800)
        self.script = ScriptModel(grid_rows=7, grid_cols=9)
        self._applying_edit = False
        self._selecting = False
        self._has_minimized = False

        # 区域计时器状态（必须在 _build_ui 之前初始化，因为 UI 信号回调可能访问）
        self._region_timer: RegionStateTimer | None = None
        self._timer_capture: WindowCapture | None = None
        self._timer_overlay: TimerOverlay | None = None
        self._timer_qtimer: QTimer | None = None
        self._timer_started = False

        # 录制器状态
        self._recorder: ActionRecorder | None = None
        self._recorder_capture: WindowCapture | None = None

        self._build_ui()
        # 应用划火柴配置并启动全局热键监听
        self._apply_matchstick_config()

    def closeEvent(self, event):
        """退出时停止划火柴全局热键监听和区域计时器。"""
        action.stop_matchstick_listener()
        self._stop_region_timer()
        event.accept()

    def _project_root(self) -> str:
        if getattr(sys, "frozen", False):
            # PyInstaller 打包后：exe 所在目录即为项目根目录
            return os.path.dirname(sys.executable)
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _config_path(self) -> str:
        return os.path.join(self._project_root(), "config.json")

    def _load_config(self):
        path = self._config_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _normalize_key_name(self, key: str) -> str:
        """把用户输入的键名规范化成 pydirectinput 能识别的名字。"""
        key = key.strip().lower()
        if key in ("", "space", "空格", " "):
            return "space"
        return key

    def _normalize_hotkey(self, key: str) -> str:
        """规范化全局热键字符串。"""
        return key.strip().lower()

    def _normalize_grid_text(self, text: str) -> str:
        """把用户输入的格子文本规范化成英文逗号分隔。"""
        return text.strip().replace("，", ",").replace(" ", "")

    def _ms_to_sf(self, ms: int):
        """把毫秒转换为 (秒, 帧)，帧范围为 0-29。"""
        s = int(ms) // 1000
        f = min(29, max(0, round((int(ms) % 1000 - 1) / 33.3)))
        return s, f

    def _ms_to_sf_for_timer(self, ms: float):
        """计时器悬浮窗专用：使用 floor 取帧，避免初始/边界帧数多 1。"""
        s = int(ms) // 1000
        f = min(29, max(0, int((ms % 1000) / 33.333)))
        return s, f

    def _sf_to_ms(self, s, f) -> int:
        """把 (秒, 帧) 转换为毫秒，补 1ms 对齐 30 帧。"""
        return int(s) * 1000 + int(round(int(f) * 33.3)) + 1

    def _game_key_set(self) -> set:
        """返回当前配置的游戏内快捷键集合。"""
        return {
            self._normalize_key_name(self.combo_pause_key.currentText()),
            self._normalize_key_name(self.line_skill_key.text()),
            self._normalize_key_name(self.line_retreat_key.text()),
        }

    def _save_config(self):
        path = self._config_path()
        config = {
            "last_script_path": self.exec_script_path.text(),
            "loop": self.chk_loop.isChecked(),
            "leak": self.chk_leak.isChecked(),
            "debug": self.chk_debug.isChecked(),
            "direct_start": self.chk_direct_start.isChecked(),
            "challenge_mode": self.chk_challenge_mode.isChecked(),
            "rec_debug": self.rec_chk_debug.isChecked(),
            "rec_stage_name": self.rec_stage_name.text(),
            "rec_stage_code": self.rec_stage_code.text(),
            "rec_grid_rows": self.rec_grid_rows.value(),
            "rec_grid_cols": self.rec_grid_cols.value(),
            "rec_operators": [self.rec_op_list.item(i).text() for i in range(self.rec_op_list.count())],
            "rec_items": [
                {"name": self.rec_item_table.item(r, 0).text(), "charges": int(self.rec_item_table.item(r, 1).text())}
                for r in range(self.rec_item_table.rowCount())
            ],
            "borrow_support": self.chk_borrow_support.isChecked(),
            "support_friend_index": self.spin_support_friend.value(),
            "support_skill": int(self.combo_support_skill.currentText()),
            "support_module": int(self.combo_support_module.currentText()),
            "pause_key": self._normalize_key_name(self.combo_pause_key.currentText()),
            "skill_key": self._normalize_key_name(self.line_skill_key.text()),
            "retreat_key": self._normalize_key_name(self.line_retreat_key.text()),
            "matchstick": {
                "select_operator": {
                    "hotkey": self._normalize_hotkey(self.line_matchstick_select.text()),
                    "enabled": self.chk_matchstick_select.isChecked(),
                },
                "pass_166ms": {
                    "hotkey": self._normalize_hotkey(self.line_matchstick_166.text()),
                    "enabled": self.chk_matchstick_166.isChecked(),
                },
                "pass_50ms": {
                    "hotkey": self._normalize_hotkey(self.line_matchstick_50.text()),
                    "enabled": self.chk_matchstick_50.isChecked(),
                },
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _apply_config(self, config: dict):
        if config.get("last_script_path"):
            self.exec_script_path.setText(config["last_script_path"])
        # 无限凸图与直接开始作战互斥，同时开启时优先关闭直接开始作战
        loop = config.get("loop", False)
        direct_start = config.get("direct_start", False)
        if loop and direct_start:
            direct_start = False
        self.chk_loop.setChecked(loop)
        self.chk_leak.setChecked(config.get("leak", False))
        self.chk_debug.setChecked(config.get("debug", False))
        self.chk_direct_start.setChecked(direct_start)
        self.chk_challenge_mode.setChecked(config.get("challenge_mode", False))
        self.rec_chk_debug.setChecked(config.get("rec_debug", False))
        self.rec_stage_name.setText(config.get("rec_stage_name", ""))
        self.rec_stage_code.setText(config.get("rec_stage_code", ""))
        self.rec_grid_rows.setValue(config.get("rec_grid_rows", 7))
        self.rec_grid_cols.setValue(config.get("rec_grid_cols", 9))
        self.rec_op_list.clear()
        for op in config.get("rec_operators", []):
            if op:
                self.rec_op_list.addItem(op)
        self.rec_item_table.setRowCount(0)
        for it in config.get("rec_items", []):
            row = self.rec_item_table.rowCount()
            self.rec_item_table.insertRow(row)
            self.rec_item_table.setItem(row, 0, QTableWidgetItem(it.get("name", "")))
            self.rec_item_table.setItem(row, 1, QTableWidgetItem(str(it.get("charges", 1))))
        self.chk_borrow_support.setChecked(config.get("borrow_support", False))
        self.spin_support_friend.setValue(config.get("support_friend_index", 0))
        skill_idx = max(0, min(2, config.get("support_skill", 1) - 1))
        self.combo_support_skill.setCurrentIndex(skill_idx)
        module_idx = max(0, min(2, config.get("support_module", 1) - 1))
        self.combo_support_module.setCurrentIndex(module_idx)

        pause_key = self._normalize_key_name(config.get("pause_key", "p"))
        idx = self.combo_pause_key.findText(pause_key)
        if idx >= 0:
            self.combo_pause_key.setCurrentIndex(idx)
        else:
            self.combo_pause_key.setCurrentText(pause_key)

        self.line_skill_key.setText(self._normalize_key_name(config.get("skill_key", "e")))
        self.line_retreat_key.setText(self._normalize_key_name(config.get("retreat_key", "q")))

        # 同步游戏内快捷键到 action 模块（GUI 进程也运行划火柴监听，需要一致）
        action.configure_keys(
            pause=self._normalize_key_name(config.get("pause_key", "p")),
            skill=self._normalize_key_name(config.get("skill_key", "e")),
            retreat=self._normalize_key_name(config.get("retreat_key", "q")),
        )

        # 划火柴热键配置
        matchstick = config.get("matchstick", {})
        for op, widget_key, widget_chk in (
            ("select_operator", self.line_matchstick_select, self.chk_matchstick_select),
            ("pass_166ms", self.line_matchstick_166, self.chk_matchstick_166),
            ("pass_50ms", self.line_matchstick_50, self.chk_matchstick_50),
        ):
            op_cfg = matchstick.get(op, {})
            widget_key.setText(self._normalize_hotkey(op_cfg.get("hotkey", {"select_operator": "r", "pass_166ms": "space", "pass_50ms": "f"}[op])))
            widget_chk.setChecked(op_cfg.get("enabled", False))

        self._on_loop_changed(self.chk_loop.checkState())
        self._on_direct_start_changed(self.chk_direct_start.checkState())
        self._on_challenge_mode_changed(self.chk_challenge_mode.checkState())
        self._on_borrow_support_changed(self.chk_borrow_support.checkState())
        # 加载完成后统一校验一次划火柴热键冲突
        self._validate_all_matchstick_hotkeys()

    def _validate_all_matchstick_hotkeys(self):
        """校验所有划火柴热键，自动禁用与游戏内快捷键冲突的热键。"""
        game_keys = self._game_key_set()
        widgets = [
            (self.line_matchstick_select, self.chk_matchstick_select, "选中干员"),
            (self.line_matchstick_166, self.chk_matchstick_166, "过 166ms"),
            (self.line_matchstick_50, self.chk_matchstick_50, "过 50ms"),
        ]
        for line, chk, name in widgets:
            hotkey = self._normalize_hotkey(line.text())
            if hotkey and "+" not in hotkey and hotkey in game_keys:
                if chk.isChecked():
                    QMessageBox.warning(
                        self,
                        "热键冲突",
                        f"划火柴热键 '{name} ({hotkey})' 与游戏内快捷键冲突，已自动禁用，请重新绑定。",
                    )
                    chk.setChecked(False)
        self._apply_matchstick_config()

    def _build_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self._build_exec_tab()
        tabs.addTab(self.exec_tab, "脚本执行")

        self._build_editor_tab()
        tabs.addTab(self.editor_tab, "脚本编辑")

        self._build_resource_tab()
        tabs.addTab(self.resource_tab, "资源更新")

        self._build_matchstick_tab()
        tabs.addTab(self.matchstick_tab, "划火柴")

        self._build_timer_tab()
        tabs.addTab(self.timer_tab, "计时器")

        self._build_recorder_tab()
        tabs.addTab(self.recorder_tab, "操作录制")

        self._build_guide_tab()
        tabs.addTab(self.guide_tab, "使用指南")

        # 所有 UI 控件创建完成后再加载配置，避免信号处理时访问未创建的控件
        self._apply_config(self._load_config())

    # ============================================================
    # 脚本执行
    # ============================================================
    def _build_exec_tab(self):
        self.exec_tab = QWidget()
        layout = QVBoxLayout(self.exec_tab)

        # 脚本选择
        script_layout = QHBoxLayout()
        script_layout.addWidget(QLabel("脚本文件"))
        self.exec_script_path = QLineEdit()
        self.exec_script_path.setPlaceholderText("选择脚本 JSON 文件...")
        script_layout.addWidget(self.exec_script_path)
        self.btn_browse = QPushButton("浏览")
        self.btn_browse.clicked.connect(self._browse_script)
        script_layout.addWidget(self.btn_browse)
        layout.addLayout(script_layout)

        # 参数勾选
        params_group = QGroupBox("运行参数")
        params_layout = QHBoxLayout(params_group)
        self.chk_loop = QCheckBox("无限凸图 (--loop)")
        self.chk_leak = QCheckBox("漏怪检测 (--leak)")
        self.chk_debug = QCheckBox("Debug (--debug)")
        self.chk_direct_start = QCheckBox("直接开始作战 (--direct-start)")
        self.chk_challenge_mode = QCheckBox("突袭模式")
        self.chk_loop.stateChanged.connect(self._on_loop_changed)
        self.chk_direct_start.stateChanged.connect(self._on_direct_start_changed)
        self.chk_challenge_mode.stateChanged.connect(self._on_challenge_mode_changed)
        params_layout.addWidget(self.chk_loop)
        params_layout.addWidget(self.chk_leak)
        params_layout.addWidget(self.chk_debug)
        params_layout.addWidget(self.chk_direct_start)
        params_layout.addWidget(self.chk_challenge_mode)
        params_layout.addStretch()
        layout.addWidget(params_group)

        # 助战参数
        support_group = QGroupBox("助战参数")
        support_layout = QHBoxLayout(support_group)
        self.chk_borrow_support = QCheckBox("借用干员")
        self.chk_borrow_support.stateChanged.connect(self._on_borrow_support_changed)
        support_layout.addWidget(self.chk_borrow_support)

        support_layout.addWidget(QLabel("好友位置"))
        self.spin_support_friend = QSpinBox()
        self.spin_support_friend.setRange(0, 8)
        self.spin_support_friend.setEnabled(False)
        support_layout.addWidget(self.spin_support_friend)

        support_layout.addWidget(QLabel("携带技能"))
        self.combo_support_skill = QComboBox()
        self.combo_support_skill.addItems(["1", "2", "3"])
        self.combo_support_skill.setEnabled(False)
        support_layout.addWidget(self.combo_support_skill)

        support_layout.addWidget(QLabel("模组选择"))
        self.combo_support_module = QComboBox()
        self.combo_support_module.addItems(["1", "2", "3"])
        self.combo_support_module.setEnabled(False)
        support_layout.addWidget(self.combo_support_module)

        support_layout.addStretch()
        layout.addWidget(support_group)

        # 键位设置
        keys_group = QGroupBox("键位设置")
        keys_layout = QHBoxLayout(keys_group)

        keys_layout.addWidget(QLabel("暂停键"))
        self.combo_pause_key = QComboBox()
        self.combo_pause_key.setEditable(True)
        self.combo_pause_key.addItems(["p", "space", "q", "e", "r", "f"])
        self.combo_pause_key.setFixedWidth(80)
        keys_layout.addWidget(self.combo_pause_key)

        keys_layout.addWidget(QLabel("技能键"))
        self.line_skill_key = QLineEdit("e")
        self.line_skill_key.setMaxLength(8)
        self.line_skill_key.setFixedWidth(60)
        keys_layout.addWidget(self.line_skill_key)

        keys_layout.addWidget(QLabel("撤退键"))
        self.line_retreat_key = QLineEdit("q")
        self.line_retreat_key.setMaxLength(8)
        self.line_retreat_key.setFixedWidth(60)
        keys_layout.addWidget(self.line_retreat_key)

        self.combo_pause_key.currentTextChanged.connect(self._on_game_key_changed)
        self.line_skill_key.textChanged.connect(self._on_game_key_changed)
        self.line_retreat_key.textChanged.connect(self._on_game_key_changed)

        keys_layout.addStretch()
        layout.addWidget(keys_group)

        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("运行脚本")
        self.btn_run.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # 状态
        self.status_label = QLabel("状态: 就绪")
        layout.addWidget(self.status_label)

        # 日志
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #d4d4d4; font-family: Consolas, monospace;"
        )
        layout.addWidget(self.log_text)

        # 绑定
        self.btn_run.clicked.connect(self._start_script)
        self.btn_stop.clicked.connect(self._stop_script)

        self.process = None

    def _browse_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择脚本", "", "JSON (*.json)")
        if path:
            self.exec_script_path.setText(path)

    def _on_borrow_support_changed(self, state):
        # stateChanged 信号传 int，checkState() 返回 Qt.CheckState 枚举，统一处理
        if isinstance(state, Qt.CheckState):
            enabled = state == Qt.CheckState.Checked
        else:
            enabled = state == Qt.CheckState.Checked.value
        # 直接开始作战模式下不启用助战参数
        if self.chk_direct_start.isChecked():
            enabled = False
        self.spin_support_friend.setEnabled(enabled)
        self.combo_support_skill.setEnabled(enabled)
        self.combo_support_module.setEnabled(enabled)

    def _on_loop_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.chk_direct_start.setChecked(False)
            self.chk_direct_start.setEnabled(False)
        else:
            self.chk_direct_start.setEnabled(True)

    def _on_direct_start_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.chk_loop.setChecked(False)
            self.chk_loop.setEnabled(False)
            self.chk_challenge_mode.setChecked(False)
            self.chk_challenge_mode.setEnabled(False)
        else:
            self.chk_loop.setEnabled(True)
            self.chk_challenge_mode.setEnabled(True)
        # 直接开始作战时禁用助战参数（已进入准备界面，无需再选助战）
        self.chk_borrow_support.setEnabled(not checked)
        self._on_borrow_support_changed(self.chk_borrow_support.checkState())

    def _on_challenge_mode_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.chk_direct_start.setChecked(False)
            self.chk_direct_start.setEnabled(False)
        else:
            self.chk_direct_start.setEnabled(True)

    def _on_game_key_changed(self, text):
        """当游戏内快捷键改变时，检查并禁用冲突的划火柴热键。"""
        game_keys = self._game_key_set()
        widgets = [
            (self.line_matchstick_select, self.chk_matchstick_select, "选中干员"),
            (self.line_matchstick_166, self.chk_matchstick_166, "过 166ms"),
            (self.line_matchstick_50, self.chk_matchstick_50, "过 50ms"),
        ]
        for line, chk, name in widgets:
            hotkey = self._normalize_hotkey(line.text())
            if hotkey and "+" not in hotkey and hotkey in game_keys and chk.isChecked():
                QMessageBox.warning(
                    self,
                    "热键冲突",
                    f"游戏内快捷键与划火柴热键 '{name} ({hotkey})' 冲突，已自动禁用该划火柴热键。",
                )
                chk.setChecked(False)
        self._apply_matchstick_config()
        # 实时同步修改后的游戏内快捷键到 action 模块
        action.configure_keys(
            pause=self._normalize_key_name(self.combo_pause_key.currentText()),
            skill=self._normalize_key_name(self.line_skill_key.text()),
            retreat=self._normalize_key_name(self.line_retreat_key.text()),
        )

    def _on_matchstick_enabled_changed(self, state):
        self._apply_matchstick_config()

    def _on_matchstick_hotkey_changed(self, text):
        sender = self.sender()
        # 找到当前修改的是哪个热键输入框
        op_map = {
            self.line_matchstick_select: ("选中干员", self.chk_matchstick_select),
            self.line_matchstick_166: ("过 166ms", self.chk_matchstick_166),
            self.line_matchstick_50: ("过 50ms", self.chk_matchstick_50),
        }
        op_name, chk = op_map.get(sender, ("", None))
        normalized = self._normalize_hotkey(text)
        game_keys = self._game_key_set()
        if normalized and "+" not in normalized and normalized in game_keys:
            QMessageBox.warning(self, "热键冲突", f"'{op_name}' 热键 '{normalized}' 与脚本键位设置中的游戏快捷键冲突，请更换。")
            # 还原为 action 模块当前配置的值
            sender.blockSignals(True)
            cfg = action.get_matchstick_config()
            sender.setText(cfg["hotkeys"].get(self._matchstick_op_from_widget(sender), ""))
            sender.blockSignals(False)
            return
        self._apply_matchstick_config()

    def _matchstick_op_from_widget(self, widget):
        if widget is self.line_matchstick_select:
            return "select_operator"
        if widget is self.line_matchstick_166:
            return "pass_166ms"
        if widget is self.line_matchstick_50:
            return "pass_50ms"
        return ""

    def _apply_matchstick_config(self):
        """同步当前 UI 中的划火柴配置到 action 模块并重启监听。"""
        hotkeys = {
            "select_operator": self._normalize_hotkey(self.line_matchstick_select.text()),
            "pass_166ms": self._normalize_hotkey(self.line_matchstick_166.text()),
            "pass_50ms": self._normalize_hotkey(self.line_matchstick_50.text()),
        }
        enabled = {
            "select_operator": self.chk_matchstick_select.isChecked(),
            "pass_166ms": self.chk_matchstick_166.isChecked(),
            "pass_50ms": self.chk_matchstick_50.isChecked(),
        }
        action.configure_matchstick(hotkeys=hotkeys, enabled=enabled)
        action.start_matchstick_listener()
        # 若计时器正在运行，实时把新的划火柴配置推过去
        if self._region_timer is not None and self._region_timer.is_running():
            matchstick_hotkeys = self._build_matchstick_hotkeys()
            self._region_timer.update_matchstick_hotkeys(
                matchstick_hotkeys if matchstick_hotkeys else None
            )
        self._save_config()

    def _start_script(self):
        script_path = self.exec_script_path.text()
        if not script_path:
            QMessageBox.warning(self, "警告", "请先选择脚本文件")
            return

        # 保存当前配置，下次启动自动恢复
        self._save_config()

        args = ["--run-script", script_path]
        if self.chk_loop.isChecked():
            args.append("--loop")
        if self.chk_leak.isChecked():
            args.append("--leak")
        if self.chk_debug.isChecked():
            args.append("--debug")
        if self.chk_direct_start.isChecked():
            args.append("--direct-start")
        if self.chk_challenge_mode.isChecked():
            args.append("--challenge-mode")
        if self.chk_borrow_support.isChecked():
            args.append("--borrow-support")
            args.extend([
                "--support-friend-index",
                str(self.spin_support_friend.value()),
                "--support-skill",
                self.combo_support_skill.currentText(),
                "--support-module",
                self.combo_support_module.currentText(),
            ])

        args.extend([
            "--pause-key", self._normalize_key_name(self.combo_pause_key.currentText()),
            "--skill-key", self._normalize_key_name(self.line_skill_key.text()),
            "--retreat-key", self._normalize_key_name(self.line_retreat_key.text()),
        ])

        self.process = QProcess()
        self.process.setWorkingDirectory(self._project_root())
        # 显式继承系统环境变量，避免某些库（如 Paddle）找不到依赖
        env = QProcessEnvironment.systemEnvironment()
        # 强制子进程使用 UTF-8 编码，防止 Windows GBK 控制台下打印特殊字符崩溃
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)
        # 将 stderr 合并到 stdout 通道，避免 C 层日志填满独立管道导致子进程阻塞
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._on_stdout)
        self.process.finished.connect(self._on_finished)

        self.log_text.clear()
        if self.chk_direct_start.isChecked():
            self.log_text.append("[系统] 直接开始作战模式，脚本初始化中...")
        else:
            self.log_text.append("[系统] 脚本初始化中，首次加载 OCR 模型可能需要十几秒...")
        # 开发环境走 python entry.py，打包后走 exe 自身
        if getattr(sys, "frozen", False):
            self.process.start(sys.executable, args)
        else:
            self.process.start(sys.executable, ["entry.py"] + args)

        self.status_label.setText("状态: 脚本初始化中...")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._has_minimized = False

    def _on_stdout(self):
        data = self.process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        lines = data.rstrip().splitlines()
        filtered = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            # 脚本实际开始运行时更新状态标签
            if "[系统] 脚本开始运行" in stripped:
                self.status_label.setText("状态: 运行中")
            # OCR 初始化成功后自动最小化窗口，避免遮挡游戏画面
            if "[OCR] 初始化成功" in stripped and not self._has_minimized:
                self._has_minimized = True
                self.showMinimized()
            # 过滤 Paddle 初始化时的进度条与模型加载噪音，以及内部状态标记
            if any(p in stripped for p in ("Creating model", "Model files already exist", "To redownload", "Loading weights", "%|", "[32m", "[0m", "[OCR] 初始化成功")):
                continue
            # 处理子进程划火柴屏蔽请求
            if stripped.startswith("__TIMER_SHIELD__:"):
                try:
                    duration_ms = float(stripped.split(":", 1)[1])
                    if self._region_timer is not None and self._region_timer.is_running():
                        self._region_timer.shield_matchstick(duration_ms)
                except Exception:
                    pass
                continue
            # 处理子进程计时器补偿请求
            if stripped.startswith("__TIMER_ADJUST__:"):
                try:
                    offset_ms = float(stripped.split(":", 1)[1])
                    if self._region_timer is not None and self._region_timer.is_running():
                        self._region_timer.adjust(offset_ms)
                except Exception:
                    pass
                continue
            filtered.append(stripped)
        if filtered:
            self.log_text.append("\n".join(filtered))

    def _on_stderr(self):
        # 兜底：把 stderr 也显示出来，方便排查子进程启动错误
        data = self.process.readAllStandardError().data().decode("utf-8", errors="replace")
        if data.strip():
            self.log_text.append(f"[stderr] {data.strip()}")

    def _on_finished(self, exit_code, exit_status):
        self.status_label.setText(f"状态: 已停止 (退出码 {exit_code})")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        action.start_matchstick_listener()
        if self._region_timer is not None and self._region_timer.is_running():
            self._region_timer.reconnect_hotkey()

    def _stop_script(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.terminate()
            QTimer.singleShot(3000, self._force_kill)

    def _force_kill(self):
        if self.process and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.log_text.append("[系统] 强制终止进程")

    # ============================================================
    # 脚本编辑
    # ============================================================
    def _build_editor_tab(self):
        self.editor_tab = QWidget()
        main_layout = QHBoxLayout(self.editor_tab)

        # 左侧：脚本基本信息
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("关卡名"))
        self.stage_name_edit = QLineEdit()
        left_panel.addWidget(self.stage_name_edit)

        left_panel.addWidget(QLabel("关卡代号"))
        self.stage_code_edit = QLineEdit()
        left_panel.addWidget(self.stage_code_edit)

        left_panel.addWidget(QLabel("地图行数"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 20)
        self.rows_spin.setValue(7)
        left_panel.addWidget(self.rows_spin)

        left_panel.addWidget(QLabel("地图列数"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 20)
        self.cols_spin.setValue(9)
        left_panel.addWidget(self.cols_spin)

        left_panel.addWidget(QLabel("部署区干员初始列表"))
        self.operators_list = QListWidget()
        self.operators_list.setMinimumHeight(200)
        self.operators_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.operators_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.operators_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.operators_list.model().rowsMoved.connect(self._sync_operators_to_script)
        left_panel.addWidget(self.operators_list)
        op_input_layout = QGridLayout()
        self.op_input = QLineEdit()
        self.op_input.setPlaceholderText("输入干员名...")
        self.btn_add_op = QPushButton("添加")
        self.btn_remove_op = QPushButton("删除")
        self.btn_up_op = QPushButton("上移")
        self.btn_down_op = QPushButton("下移")
        op_input_layout.addWidget(self.op_input, 0, 0, 1, 2)
        op_input_layout.addWidget(self.btn_add_op, 1, 0)
        op_input_layout.addWidget(self.btn_remove_op, 1, 1)
        op_input_layout.addWidget(self.btn_up_op, 2, 0)
        op_input_layout.addWidget(self.btn_down_op, 2, 1)
        left_panel.addLayout(op_input_layout)

        left_panel.addWidget(QLabel("部署区道具初始列表"))
        self.items_table = QTableWidget()
        self.items_table.setColumnCount(2)
        self.items_table.setHorizontalHeaderLabels(["道具名", "次数"])
        self.items_table.setMinimumHeight(140)
        self.items_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.items_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.items_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.items_table.setDragDropOverwriteMode(False)
        self.items_table.model().rowsMoved.connect(self._sync_items_to_script)
        self.items_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.items_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        left_panel.addWidget(self.items_table)
        item_input_layout = QHBoxLayout()
        self.item_input = QLineEdit()
        self.item_input.setPlaceholderText("道具名...")
        self.item_charges_input = QSpinBox()
        self.item_charges_input.setRange(1, 999)
        self.item_charges_input.setValue(1)
        item_input_layout.addWidget(self.item_input)
        item_input_layout.addWidget(self.item_charges_input)
        left_panel.addLayout(item_input_layout)

        item_btn_layout = QGridLayout()
        self.btn_add_item = QPushButton("添加")
        self.btn_remove_item = QPushButton("删除")
        self.btn_up_item = QPushButton("上移")
        self.btn_down_item = QPushButton("下移")
        item_btn_layout.addWidget(self.btn_add_item, 0, 0)
        item_btn_layout.addWidget(self.btn_remove_item, 0, 1)
        item_btn_layout.addWidget(self.btn_up_item, 1, 0)
        item_btn_layout.addWidget(self.btn_down_item, 1, 1)
        left_panel.addLayout(item_btn_layout)

        left_panel.addStretch()

        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMaximumWidth(220)
        main_layout.addWidget(left_widget, 0)

        # 中间：时间轴列表
        mid_panel = QVBoxLayout()
        mid_panel.addWidget(QLabel("时间轴操作"))
        self.action_table = QTableWidget()
        self.action_table.setColumnCount(7)
        self.action_table.setHorizontalHeaderLabels(
            ["秒", "帧", "操作", "干员/道具", "格子", "方向", "装置"]
        )
        self.action_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.action_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.DoubleClicked
        )
        mid_panel.addWidget(self.action_table)

        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("添加")
        self.btn_remove = QPushButton("删除")
        self.btn_up = QPushButton("上移")
        self.btn_down = QPushButton("下移")
        btn_layout.addWidget(self.btn_add)
        btn_layout.addWidget(self.btn_remove)
        btn_layout.addWidget(self.btn_up)
        btn_layout.addWidget(self.btn_down)
        mid_panel.addLayout(btn_layout)

        main_layout.addLayout(mid_panel, 3)

        # 右侧：操作详情
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("操作属性"))

        # 时间：秒 + 帧（30fps，每帧 33.3ms，补 1ms 对齐）
        right_panel.addWidget(QLabel("时间"))
        time_layout = QHBoxLayout()
        self.edit_time_s = QSpinBox()
        self.edit_time_s.setRange(0, 999999)
        self.edit_time_s.setSuffix("秒")
        self.edit_time_s.setToolTip("秒")
        time_layout.addWidget(self.edit_time_s)
        self.edit_time_f = QSpinBox()
        self.edit_time_f.setRange(0, 29)
        self.edit_time_f.setSuffix("帧")
        self.edit_time_f.setToolTip("帧 (0-29)")
        time_layout.addWidget(self.edit_time_f)
        time_layout.addStretch()
        right_panel.addLayout(time_layout)

        # 操作类型显示映射（枚举值保持英文，UI 显示中文）
        self._action_labels = {
            ActionType.DEPLOY: "部署",
            ActionType.RETREAT: "撤退",
            ActionType.SKILL: "技能",
            ActionType.SPEED_UP: "加速",
            ActionType.SPEED_DOWN: "减速",
            ActionType.PAUSE: "暂停",
            ActionType.ADD_ITEM: "部署区新增道具",
        }
        self._action_labels_rev = {v: k for k, v in self._action_labels.items()}

        right_panel.addWidget(QLabel("操作类型"))
        self.combo_action = QComboBox()
        for act in ActionType:
            self.combo_action.addItem(self._action_labels.get(act, act.value), act)
        right_panel.addWidget(self.combo_action)

        right_panel.addWidget(QLabel("干员/道具名称"))
        self.combo_op = QComboBox()
        self.combo_op.setEditable(True)
        right_panel.addWidget(self.combo_op)

        # 格子输入（通用）
        self.grid_input_widget = QWidget()
        grid_layout = QVBoxLayout(self.grid_input_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(QLabel("格子 (行,列)"))
        self.edit_grid = QLineEdit()
        grid_layout.addWidget(self.edit_grid)
        right_panel.addWidget(self.grid_input_widget)

        # 新增道具专用输入（序号 + 数量）
        self.item_index_widget = QWidget()
        item_layout = QVBoxLayout(self.item_index_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.addWidget(QLabel("序号"))
        self.edit_item_index = QSpinBox()
        self.edit_item_index.setRange(0, 999)
        item_layout.addWidget(self.edit_item_index)
        item_layout.addWidget(QLabel("数量"))
        self.edit_item_charges = QSpinBox()
        self.edit_item_charges.setRange(1, 999)
        item_layout.addWidget(self.edit_item_charges)
        self.item_index_widget.hide()
        right_panel.addWidget(self.item_index_widget)

        right_panel.addWidget(QLabel("方向"))
        self.edit_dir = QComboBox()
        self.edit_dir.addItems(["", "up", "down", "left", "right"])
        right_panel.addWidget(self.edit_dir)

        right_panel.addWidget(QLabel("是否为场上装置"))
        self.chk_is_object = QCheckBox("is_object")
        right_panel.addWidget(self.chk_is_object)

        right_panel.addStretch()

        right_top = QWidget()
        right_top.setLayout(right_panel)

        # 脚本管理区域（固定在右侧底部）
        script_mgmt = QGroupBox("脚本管理")
        mgmt_layout = QVBoxLayout(script_mgmt)
        self.btn_new = QPushButton("新建")
        self.btn_open = QPushButton("打开")
        self.btn_save = QPushButton("保存")
        mgmt_layout.addWidget(self.btn_new)
        mgmt_layout.addWidget(self.btn_open)
        mgmt_layout.addWidget(self.btn_save)

        right_container = QVBoxLayout()
        right_container.addWidget(right_top, 1)
        right_container.addWidget(script_mgmt, 0)

        right_widget = QWidget()
        right_widget.setLayout(right_container)
        right_widget.setMaximumWidth(360)
        main_layout.addWidget(right_widget, 0)

        # 绑定事件
        self.btn_add.clicked.connect(self._add_action)
        self.btn_remove.clicked.connect(self._remove_action)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down.clicked.connect(self._move_down)
        self.btn_new.clicked.connect(self._new_script)
        self.btn_open.clicked.connect(self._open_script)
        self.btn_save.clicked.connect(self._save_script)
        self.action_table.itemSelectionChanged.connect(self._on_select)
        self.action_table.cellChanged.connect(self._on_cell_changed)
        self.combo_action.currentTextChanged.connect(self._on_action_type_changed)
        self.rows_spin.valueChanged.connect(self._update_script_meta)
        self.cols_spin.valueChanged.connect(self._update_script_meta)
        self.stage_name_edit.textChanged.connect(self._update_script_meta)
        self.stage_code_edit.textChanged.connect(self._update_script_meta)
        self.btn_add_op.clicked.connect(self._add_operator)
        self.btn_remove_op.clicked.connect(self._remove_operator)
        self.btn_up_op.clicked.connect(self._move_op_up)
        self.btn_down_op.clicked.connect(self._move_op_down)
        self.btn_add_item.clicked.connect(self._add_item)
        self.btn_remove_item.clicked.connect(self._remove_item)
        self.btn_up_item.clicked.connect(self._move_item_up)
        self.btn_down_item.clicked.connect(self._move_item_down)
        self.items_table.cellChanged.connect(self._on_item_cell_changed)

        # 右侧操作属性变更后自动同步到当前时间轴条目，无需按“应用修改”
        self.edit_time_s.valueChanged.connect(self._auto_apply_edit)
        self.edit_time_f.valueChanged.connect(self._auto_apply_edit)
        self.combo_action.currentIndexChanged.connect(self._auto_apply_edit)
        self.combo_op.currentTextChanged.connect(self._auto_apply_edit)
        self.edit_grid.textChanged.connect(self._auto_apply_edit)
        self.edit_item_index.valueChanged.connect(self._auto_apply_edit)
        self.edit_item_charges.valueChanged.connect(self._auto_apply_edit)
        self.edit_dir.currentTextChanged.connect(self._auto_apply_edit)
        self.chk_is_object.stateChanged.connect(self._auto_apply_edit)

        self._refresh_table()

    def _update_script_meta(self):
        self.script.stage_name = self.stage_name_edit.text() or None
        self.script.stage_code = self.stage_code_edit.text() or None
        self.script.grid_rows = self.rows_spin.value()
        self.script.grid_cols = self.cols_spin.value()

    def _sync_operators_to_script(self):
        self.script.operators = []
        for i in range(self.operators_list.count()):
            text = self.operators_list.item(i).text().strip()
            if text:
                self.script.operators.append(text)
        self._refresh_op_combo()

    def _sync_items_to_script(self):
        self.script.items = []
        for i in range(self.items_table.rowCount()):
            name_item = self.items_table.item(i, 0)
            charges_item = self.items_table.item(i, 1)
            if name_item:
                name = name_item.text().strip()
                charges = 1
                if charges_item:
                    try:
                        charges = int(charges_item.text())
                    except ValueError:
                        charges = 1
                if name:
                    self.script.items.append(ItemInfo(name=name, charges=charges))
        self._refresh_op_combo()

    def _add_operator(self):
        name = self.op_input.text().strip()
        if name:
            self.operators_list.addItem(name)
            self.op_input.clear()
            self._sync_operators_to_script()

    def _remove_operator(self):
        idx = self.operators_list.currentRow()
        if idx >= 0:
            self.operators_list.takeItem(idx)
            self._sync_operators_to_script()

    def _move_op_up(self):
        idx = self.operators_list.currentRow()
        if idx > 0:
            item = self.operators_list.takeItem(idx)
            self.operators_list.insertItem(idx - 1, item)
            self.operators_list.setCurrentRow(idx - 1)
            self._sync_operators_to_script()

    def _move_op_down(self):
        idx = self.operators_list.currentRow()
        if 0 <= idx < self.operators_list.count() - 1:
            item = self.operators_list.takeItem(idx)
            self.operators_list.insertItem(idx + 1, item)
            self.operators_list.setCurrentRow(idx + 1)
            self._sync_operators_to_script()

    def _add_item(self):
        name = self.item_input.text().strip()
        charges = self.item_charges_input.value()
        if name:
            row = self.items_table.rowCount()
            self.items_table.insertRow(row)
            self.items_table.setItem(row, 0, QTableWidgetItem(name))
            self.items_table.setItem(row, 1, QTableWidgetItem(str(charges)))
            self.item_input.clear()
            self.item_charges_input.setValue(1)
            self._sync_items_to_script()

    def _remove_item(self):
        idx = self.items_table.currentRow()
        if idx >= 0:
            self.items_table.removeRow(idx)
            self._sync_items_to_script()

    def _move_item_up(self):
        idx = self.items_table.currentRow()
        if idx > 0:
            self._swap_item_rows(idx, idx - 1)
            self.items_table.selectRow(idx - 1)
            self._sync_items_to_script()

    def _move_item_down(self):
        idx = self.items_table.currentRow()
        if 0 <= idx < self.items_table.rowCount() - 1:
            self._swap_item_rows(idx, idx + 1)
            self.items_table.selectRow(idx + 1)
            self._sync_items_to_script()

    def _swap_item_rows(self, i: int, j: int):
        """交换道具表中两行的内容（不触发 cellChanged 信号循环）。"""
        self.items_table.blockSignals(True)
        for col in range(self.items_table.columnCount()):
            item_i = self.items_table.takeItem(i, col)
            item_j = self.items_table.takeItem(j, col)
            self.items_table.setItem(i, col, item_j)
            self.items_table.setItem(j, col, item_i)
        self.items_table.blockSignals(False)

    def _on_item_cell_changed(self, row, col):
        self._sync_items_to_script()

    def _refresh_op_combo(self):
        current = self.combo_op.currentText()
        self.combo_op.blockSignals(True)
        self.combo_op.clear()
        for op in self.script.operators:
            self.combo_op.addItem(op)
        if self.script.items:
            for item in self.script.items:
                self.combo_op.addItem(item.name)
        idx = self.combo_op.findText(current)
        if idx >= 0:
            self.combo_op.setCurrentIndex(idx)
        else:
            self.combo_op.setEditText(current)
        self.combo_op.blockSignals(False)

    def _refresh_table(self):
        # 阻断 cellChanged 避免刷新时触发循环
        self.action_table.blockSignals(True)
        # 记录当前选中的操作特征，排序后尽量恢复选中
        selected_time = None
        selected_action = None
        selected_op = None
        idx = self.action_table.currentRow()
        if 0 <= idx < len(self.script.actions):
            selected_time = self.script.actions[idx].time_ms
            selected_action = self.script.actions[idx].action
            selected_op = self.script.actions[idx].operator_name

        self.script.sort_actions()
        self.action_table.setRowCount(len(self.script.actions))
        new_idx = -1
        for i, act in enumerate(self.script.actions):
            s, f = self._ms_to_sf(act.time_ms)
            self.action_table.setItem(i, 0, QTableWidgetItem(str(s)))
            self.action_table.setItem(i, 1, QTableWidgetItem(str(f)))
            action_text = self._action_labels.get(act.action, act.action.value if act.action else "")
            self.action_table.setItem(i, 2, QTableWidgetItem(action_text))
            self.action_table.setItem(i, 3, QTableWidgetItem(act.operator_name or ""))
            grid = act.grid
            if (
                grid is None
                and act.action in (ActionType.RETREAT, ActionType.SKILL)
                and act.operator_name
            ):
                grid = self._get_default_grid_for_action(i, act.operator_name)
            try:
                grid_str = f"{grid[0]},{grid[1]}" if grid else ""
            except Exception:
                grid_str = ""
            self.action_table.setItem(i, 4, QTableWidgetItem(grid_str))
            self.action_table.setItem(i, 5, QTableWidgetItem(act.direction or ""))
            self.action_table.setItem(i, 6, QTableWidgetItem("是" if act.is_object else ""))
            if (
                act.time_ms == selected_time
                and act.action == selected_action
                and act.operator_name == selected_op
            ):
                new_idx = i

        self.action_table.blockSignals(False)
        if new_idx >= 0:
            self.action_table.selectRow(new_idx)

    def _get_default_grid_for_action(self, action_idx: int, operator_name: str) -> Optional[Tuple[int, int]]:
        """从当前 action 向前查找同一干员最近的 DEPLOY 格子作为默认值。"""
        for prev_act in reversed(self.script.actions[:action_idx]):
            if (
                prev_act.action == ActionType.DEPLOY
                and prev_act.operator_name == operator_name
            ):
                return prev_act.grid
        return None

    def _on_select(self):
        if self._applying_edit:
            return
        self._selecting = True
        try:
            idx = self.action_table.currentRow()
            if 0 <= idx < len(self.script.actions):
                act = self.script.actions[idx]
                s, f = self._ms_to_sf(act.time_ms)
                self.edit_time_s.setValue(s)
                self.edit_time_f.setValue(f)
                self.combo_action.setCurrentText(self._action_labels.get(act.action, act.action.value if act.action else ""))
                self.combo_op.setEditText(act.operator_name or "")
                if act.action == ActionType.ADD_ITEM:
                    if act.grid:
                        self.edit_item_index.setValue(act.grid[0])
                        self.edit_item_charges.setValue(act.grid[1])
                    else:
                        self.edit_item_index.setValue(0)
                        self.edit_item_charges.setValue(1)
                else:
                    grid = act.grid
                    if (
                        grid is None
                        and act.action in (ActionType.RETREAT, ActionType.SKILL)
                        and act.operator_name
                    ):
                        grid = self._get_default_grid_for_action(idx, act.operator_name)
                    self.edit_grid.setText(f"{grid[0]},{grid[1]}" if grid else "")
                self.edit_dir.setCurrentText(act.direction or "")
                self.chk_is_object.setChecked(act.is_object)
                self._on_action_type_changed()
        finally:
            self._selecting = False

    def _on_cell_changed(self, row, col):
        """在时间轴表格中直接编辑后同步回数据与右侧面板。"""
        if not (0 <= row < len(self.script.actions)):
            return
        act = self.script.actions[row]
        val = self.action_table.item(row, col).text().strip()

        if col in (0, 1):  # 秒 / 帧
            try:
                s = int(self.action_table.item(row, 0).text().strip())
                f = int(self.action_table.item(row, 1).text().strip())
                act.time_ms = self._sf_to_ms(s, f)
            except ValueError:
                pass
        elif col == 2:  # 操作类型
            # 表格中显示的是中文，需要反向映射回枚举值
            mapped = self._action_labels_rev.get(val, val)
            try:
                act.action = ActionType(mapped)
            except ValueError:
                pass
        elif col == 3:  # 干员
            act.operator_name = val or None
        elif col == 4:  # 格子
            val = self._normalize_grid_text(val)
            if val:
                try:
                    r, c = map(int, val.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
        elif col == 5:  # 方向
            act.direction = val or None
        elif col == 6:  # 道具
            act.is_object = val == "是"

        # 只有修改时间列时才需要重新排序并全量刷新；其他列直接同步右侧面板即可
        if col in (0, 1):
            self._refresh_table()
        elif self.action_table.currentRow() == row:
            self._on_select()

    def _on_action_type_changed(self):
        act = self.combo_action.currentData()
        if act == ActionType.DEPLOY:
            self.grid_input_widget.show()
            self.item_index_widget.hide()
            self.edit_dir.setEnabled(True)
            self.combo_op.setEnabled(True)
            self.chk_is_object.setEnabled(True)
        elif act in (ActionType.RETREAT, ActionType.SKILL):
            self.grid_input_widget.show()
            self.item_index_widget.hide()
            self.edit_dir.setEnabled(False)
            self.combo_op.setEnabled(True)
            self.chk_is_object.setEnabled(True)
        elif act == ActionType.ADD_ITEM:
            self.grid_input_widget.hide()
            self.item_index_widget.show()
            self.edit_dir.setEnabled(False)
            self.combo_op.setEnabled(True)
            self.chk_is_object.setEnabled(False)
        else:  # SPEED_UP, SPEED_DOWN, PAUSE
            self.grid_input_widget.show()
            self.item_index_widget.hide()
            self.edit_dir.setEnabled(False)
            self.combo_op.setEnabled(False)
            self.chk_is_object.setEnabled(False)

    def _apply_edit(self):
        idx = self.action_table.currentRow()
        if not (0 <= idx < len(self.script.actions)):
            return
        act = self.script.actions[idx]
        act.time_ms = self._sf_to_ms(self.edit_time_s.value(), self.edit_time_f.value())
        act.action = self.combo_action.currentData()

        if act.action == ActionType.DEPLOY:
            act.operator_name = self.combo_op.currentText() or None
            grid_text = self._normalize_grid_text(self.edit_grid.text())
            if grid_text:
                try:
                    r, c = map(int, grid_text.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
            act.direction = self.edit_dir.currentText() or None
            act.is_object = self.chk_is_object.isChecked()
        elif act.action in (ActionType.RETREAT, ActionType.SKILL):
            act.operator_name = self.combo_op.currentText() or None
            grid_text = self._normalize_grid_text(self.edit_grid.text())
            if grid_text:
                try:
                    r, c = map(int, grid_text.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
            act.direction = None
            act.is_object = self.chk_is_object.isChecked()
        elif act.action == ActionType.ADD_ITEM:
            act.operator_name = self.combo_op.currentText() or None
            act.grid = (self.edit_item_index.value(), self.edit_item_charges.value())
            act.direction = None
            act.is_object = False
        else:
            act.operator_name = None
            act.grid = None
            act.direction = None
            act.is_object = False

        self._refresh_table()

    def _auto_apply_edit(self):
        """右侧属性变更时自动应用到当前选中的时间轴条目。"""
        if self._applying_edit or self._selecting:
            return
        self._applying_edit = True
        try:
            self._apply_edit()
        finally:
            self._applying_edit = False

    def _new_script(self):
        self.script = ScriptModel(grid_rows=7, grid_cols=9)
        for w in (self.stage_name_edit, self.stage_code_edit, self.rows_spin, self.cols_spin):
            w.blockSignals(True)
        self.stage_name_edit.clear()
        self.stage_code_edit.clear()
        self.rows_spin.setValue(7)
        self.cols_spin.setValue(9)
        for w in (self.stage_name_edit, self.stage_code_edit, self.rows_spin, self.cols_spin):
            w.blockSignals(False)
        self.operators_list.clear()
        self.items_table.setRowCount(0)
        self._refresh_op_combo()
        self._refresh_table()

    def _open_script(self):
        path, _ = QFileDialog.getOpenFileName(self, "打开脚本", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.script = ScriptModel(**data)
        except Exception as e:
            QMessageBox.critical(self, "打开失败", f"脚本格式错误或解析失败:\n{e}")
            return

        for w in (self.stage_name_edit, self.stage_code_edit, self.rows_spin, self.cols_spin):
            w.blockSignals(True)
        self.stage_name_edit.setText(self.script.stage_name or "")
        self.stage_code_edit.setText(self.script.stage_code or "")
        self.rows_spin.setValue(self.script.grid_rows)
        self.cols_spin.setValue(self.script.grid_cols)
        for w in (self.stage_name_edit, self.stage_code_edit, self.rows_spin, self.cols_spin):
            w.blockSignals(False)

        self.operators_list.clear()
        for op in self.script.operators:
            self.operators_list.addItem(op)

        self.items_table.blockSignals(True)
        self.items_table.setRowCount(0)
        for item in self.script.items:
            row = self.items_table.rowCount()
            self.items_table.insertRow(row)
            self.items_table.setItem(row, 0, QTableWidgetItem(item.name))
            self.items_table.setItem(row, 1, QTableWidgetItem(str(item.charges)))
        self.items_table.blockSignals(False)
        self._refresh_op_combo()

        self._refresh_table()

    def _save_script(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存脚本", "", "JSON (*.json)")
        if path:
            if not path.endswith(".json"):
                path += ".json"
            self.script.sort_actions()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.script.model_dump(), f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存成功", f"已保存到:\n{path}")

    def _add_action(self):
        # 先保存当前正在编辑的内容，避免丢失
        self._apply_edit()
        # 默认值取当前最大时间，方便用户在末尾追加事件
        max_time_ms = max((a.time_ms for a in self.script.actions), default=0)
        act = OperatorAction(time_ms=max_time_ms, action=ActionType.DEPLOY)
        self.script.actions.append(act)
        self._refresh_table()
        new_idx = len(self.script.actions) - 1
        self.action_table.selectRow(new_idx)
        self.action_table.scrollToItem(
            self.action_table.item(new_idx, 0),
            QAbstractItemView.ScrollHint.EnsureVisible,
        )

    def _remove_action(self):
        self._apply_edit()
        idx = self.action_table.currentRow()
        if 0 <= idx < len(self.script.actions):
            del self.script.actions[idx]
            self._refresh_table()
            if idx < len(self.script.actions):
                self.action_table.selectRow(idx)
            elif self.script.actions:
                self.action_table.selectRow(len(self.script.actions) - 1)

    def _move_up(self):
        self._apply_edit()
        idx = self.action_table.currentRow()
        if idx > 0:
            self.script.actions[idx], self.script.actions[idx - 1] = (
                self.script.actions[idx - 1],
                self.script.actions[idx],
            )
            self._refresh_table()
            self.action_table.selectRow(idx - 1)

    def _move_down(self):
        self._apply_edit()
        idx = self.action_table.currentRow()
        if 0 <= idx < len(self.script.actions) - 1:
            self.script.actions[idx], self.script.actions[idx + 1] = (
                self.script.actions[idx + 1],
                self.script.actions[idx],
            )
            self._refresh_table()
            self.action_table.selectRow(idx + 1)

    # ============================================================
    # 资源更新
    # ============================================================
    def _build_resource_tab(self):
        self.resource_tab = QWidget()
        layout = QVBoxLayout(self.resource_tab)

        layout.addWidget(QLabel("levels.json 资源更新"))
        layout.addWidget(
            QLabel("选择新的 levels.json 文件，点击更新后将会覆盖 core/resource/levels.json")
        )

        file_layout = QHBoxLayout()
        self.resource_path = QLineEdit()
        self.resource_path.setPlaceholderText("选择 levels.json 文件...")
        file_layout.addWidget(self.resource_path)
        self.btn_resource_browse = QPushButton("浏览")
        self.btn_resource_browse.clicked.connect(self._browse_resource)
        file_layout.addWidget(self.btn_resource_browse)
        layout.addLayout(file_layout)

        self.btn_update_resource = QPushButton("更新资源")
        self.btn_update_resource.clicked.connect(self._update_resource)
        layout.addWidget(self.btn_update_resource)

        self.resource_status = QLabel("状态: 未更新")
        layout.addWidget(self.resource_status)

        layout.addStretch()

    def _browse_resource(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 levels.json", "", "JSON (*.json)")
        if path:
            self.resource_path.setText(path)

    def _update_resource(self):
        src = self.resource_path.text()
        if not src:
            QMessageBox.warning(self, "警告", "请先选择 levels.json 文件")
            return
        dst = os.path.join(self._project_root(), "core", "resource", "levels.json")
        try:
            shutil.copy2(src, dst)
            self.resource_status.setText(f"状态: 更新成功 -> {dst}")
            QMessageBox.information(self, "成功", f"已更新 levels.json\n目标: {dst}")
        except Exception as e:
            self.resource_status.setText(f"状态: 更新失败 - {e}")
            QMessageBox.critical(self, "错误", f"更新失败: {e}")

    # ============================================================
    # 划火柴
    # ============================================================
    def _build_matchstick_tab(self):
        self.matchstick_tab = QWidget()
        layout = QVBoxLayout(self.matchstick_tab)

        # 操作行：选中干员
        row_select = QHBoxLayout()
        row_select.addWidget(QLabel("选中干员"))
        self.chk_matchstick_select = QCheckBox("启用")
        self.chk_matchstick_select.stateChanged.connect(self._on_matchstick_enabled_changed)
        row_select.addWidget(self.chk_matchstick_select)
        self.line_matchstick_select = QLineEdit("r")
        self.line_matchstick_select.setMaxLength(16)
        self.line_matchstick_select.setFixedWidth(100)
        self.line_matchstick_select.textChanged.connect(self._on_matchstick_hotkey_changed)
        row_select.addWidget(self.line_matchstick_select)
        row_select.addStretch()
        layout.addLayout(row_select)

        # 操作行：过 166ms
        row_166 = QHBoxLayout()
        row_166.addWidget(QLabel("过 166ms"))
        self.chk_matchstick_166 = QCheckBox("启用")
        self.chk_matchstick_166.stateChanged.connect(self._on_matchstick_enabled_changed)
        row_166.addWidget(self.chk_matchstick_166)
        self.line_matchstick_166 = QLineEdit("space")
        self.line_matchstick_166.setMaxLength(16)
        self.line_matchstick_166.setFixedWidth(100)
        self.line_matchstick_166.textChanged.connect(self._on_matchstick_hotkey_changed)
        row_166.addWidget(self.line_matchstick_166)
        row_166.addStretch()
        layout.addLayout(row_166)

        # 操作行：过 50ms
        row_50 = QHBoxLayout()
        row_50.addWidget(QLabel("过 50ms"))
        self.chk_matchstick_50 = QCheckBox("启用")
        self.chk_matchstick_50.stateChanged.connect(self._on_matchstick_enabled_changed)
        row_50.addWidget(self.chk_matchstick_50)
        self.line_matchstick_50 = QLineEdit("f")
        self.line_matchstick_50.setMaxLength(16)
        self.line_matchstick_50.setFixedWidth(100)
        self.line_matchstick_50.textChanged.connect(self._on_matchstick_hotkey_changed)
        row_50.addWidget(self.line_matchstick_50)
        row_50.addStretch()
        layout.addLayout(row_50)

        # 说明文本
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml(
            """
            <h3>划火柴快捷键</h3>
            <p>在此绑定全局热键，开启后可在游戏中快速执行划火柴操作，为了减小电脑端误差，所有操作请在子弹时间下进行，否则不是刚好过1帧和0帧选取。</p>
            <ul>
                <li><b>选中干员</b>：用于选中鼠标指向的干员/装置。</li>
                <li><b>过 166ms</b>： 推进166ms，子弹时间下即为33ms，用于在子弹时间中推进一帧。</li>
                <li><b>过 50ms</b>：推进50ms，子弹时间下即为10ms，补齐不到1帧的时间。</li>
            </ul>
            <p><span style="color: red;">热键不可与脚本执行页中的暂停/技能/撤退键相同，否则会造成循环触发。</span></p>
            """
        )
        layout.addWidget(info)
        layout.addStretch()

    # ============================================================
    # 区域计时器
    # ============================================================
    def _build_timer_tab(self):
        self.timer_tab = QWidget()
        layout = QVBoxLayout(self.timer_tab)

        layout.addWidget(QLabel("基于游戏内区域状态的悬浮计时器"))
        layout.addWidget(
            QLabel(
                "暂停区域 : 控制计时/暂停 | "
                "倍速区域 : 控制倍率 1.0x / 0.2x"
            )
        )

        btn_layout = QHBoxLayout()
        self.btn_timer_start = QPushButton("开始计时")
        self.btn_timer_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_timer_start.clicked.connect(self._start_region_timer)
        btn_layout.addWidget(self.btn_timer_start)

        self.btn_timer_stop = QPushButton("停止计时")
        self.btn_timer_stop.setEnabled(False)
        self.btn_timer_stop.clicked.connect(self._stop_region_timer)
        btn_layout.addWidget(self.btn_timer_stop)

        self.btn_timer_reset = QPushButton("重置时间")
        self.btn_timer_reset.clicked.connect(self._reset_region_timer)
        btn_layout.addWidget(self.btn_timer_reset)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.chk_timer_debug = QCheckBox("Debug 输出")
        layout.addWidget(self.chk_timer_debug)

        self.timer_status = QLabel("状态: 就绪")
        layout.addWidget(self.timer_status)

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
        """根据 action 模块当前配置构建传给 RegionStateTimer 的划火柴热键字典。"""
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
        if self._region_timer is not None and self._region_timer.is_running():
            return

        try:
            self._timer_capture = WindowCapture(backend="mss")
            matchstick_hotkeys = self._build_matchstick_hotkeys()

            self._region_timer = RegionStateTimer(
                self._timer_capture,
                pause_key=action.pause_key(),
                debug=self.chk_timer_debug.isChecked(),
                matchstick_hotkeys=matchstick_hotkeys if matchstick_hotkeys else None,
            )
            self._region_timer.start(use_cost_detection=True)
        except Exception as e:
            QMessageBox.critical(self, "启动失败", f"计时器初始化失败:\n{e}")
            return

        self._timer_overlay = TimerOverlay(
            on_pause_clicked=self._toggle_timer_pause,
            on_reset_clicked=self._reset_region_timer,
        )
        self._timer_overlay.show()
        self.showMinimized()

        self._timer_qtimer = QTimer(self)
        self._timer_qtimer.setTimerType(Qt.TimerType.PreciseTimer)
        self._timer_qtimer.timeout.connect(self._on_timer_tick)
        self._timer_qtimer.start(20)

        self._timer_started = False
        self.btn_timer_start.setEnabled(False)
        self.btn_timer_stop.setEnabled(True)
        self.timer_status.setText("状态: 等待初始状态...")

    def _on_timer_tick(self):
        if self._region_timer is None:
            return
        info = self._region_timer.tick()
        if not info.get("started"):
            self.timer_status.setText(
                f"状态: 等待初始状态 | A={info.get('count_a')} B={info.get('count_b')}"
            )
            return

        elapsed = info["elapsed_ms"]
        s, f = self._ms_to_sf_for_timer(elapsed)
        self._timer_overlay.update_time(
            elapsed, s, f, info["rate"], info["paused"]
        )
        self._timer_overlay.set_pause_text(self._region_timer.is_manual_paused())
        self.timer_status.setText(
            f"状态: 运行中 | {s}s {f:02d}f | rate={info['rate']} | paused={info['paused']}"
        )

    def _toggle_timer_pause(self):
        if self._region_timer is None:
            return
        self._region_timer.toggle_manual_pause()
        is_paused = self._region_timer.is_manual_paused()
        if self._timer_overlay is not None:
            self._timer_overlay.set_pause_text(is_paused)
        self.timer_status.setText(f"状态: {'已手动暂停' if is_paused else '运行中'}")

    def _stop_region_timer(self):
        if self._timer_qtimer is not None:
            self._timer_qtimer.stop()
            self._timer_qtimer = None
        if self._region_timer is not None:
            self._region_timer.stop()
            self._region_timer = None
        if self._timer_overlay is not None:
            self._timer_overlay.close()
            self._timer_overlay = None
        self._timer_capture = None
        self._timer_started = False
        self.btn_timer_start.setEnabled(True)
        self.btn_timer_stop.setEnabled(False)
        self.timer_status.setText("状态: 已停止")
        if self.isMinimized():
            self.showNormal()

    def _reset_region_timer(self):
        if self._region_timer is None:
            return
        # 先暂停 QTimer，避免重置过程中被 tick 打断导致跳帧
        if self._timer_qtimer is not None:
            self._timer_qtimer.stop()
        # 重置为初始等待状态，并自动暂停，等待用户手动继续
        self._region_timer.start()
        self._region_timer.manual_pause()
        if self._timer_overlay is not None:
            self._timer_overlay.update_time(0.0, 0,
                                            0, 1.0, True)
            self._timer_overlay.set_pause_text(True)
        self.timer_status.setText("状态: 已重置并暂停，点击继续后开始计时")
        # 恢复 QTimer
        if self._timer_qtimer is not None:
            self._timer_qtimer.start(20)

    # ============================================================
    # 操作录制
    # ============================================================
    def _build_recorder_tab(self):
        self.recorder_tab = QWidget()
        layout = QVBoxLayout(self.recorder_tab)

        # 参数区域
        param_group = QGroupBox("录制参数")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("关卡名称:"), 0, 0)
        self.rec_stage_name = QLineEdit()
        self.rec_stage_name.setPlaceholderText("如 1-7")
        param_layout.addWidget(self.rec_stage_name, 0, 1)

        param_layout.addWidget(QLabel("关卡代号:"), 0, 2)
        self.rec_stage_code = QLineEdit()
        param_layout.addWidget(self.rec_stage_code, 0, 3)

        param_layout.addWidget(QLabel("地图行数:"), 1, 0)
        self.rec_grid_rows = QSpinBox()
        self.rec_grid_rows.setRange(1, 50)
        self.rec_grid_rows.setValue(7)
        param_layout.addWidget(self.rec_grid_rows, 1, 1)

        param_layout.addWidget(QLabel("地图列数:"), 1, 2)
        self.rec_grid_cols = QSpinBox()
        self.rec_grid_cols.setRange(1, 50)
        self.rec_grid_cols.setValue(9)
        param_layout.addWidget(self.rec_grid_cols, 1, 3)

        self.rec_chk_debug = QCheckBox("Debug 模式")
        param_layout.addWidget(self.rec_chk_debug, 2, 0, 1, 4)

        # 干员列表 + 道具列表并排
        op_group = QGroupBox("干员列表")
        op_layout = QVBoxLayout(op_group)
        self.rec_op_list = QListWidget()
        self.rec_op_list.setMinimumHeight(100)
        op_layout.addWidget(self.rec_op_list)
        op_input_layout = QHBoxLayout()
        self.rec_op_input = QLineEdit()
        self.rec_op_input.setPlaceholderText("输入干员名...")
        self.rec_op_add_btn = QPushButton("添加")
        self.rec_op_remove_btn = QPushButton("删除")
        op_input_layout.addWidget(self.rec_op_input)
        op_input_layout.addWidget(self.rec_op_add_btn)
        op_input_layout.addWidget(self.rec_op_remove_btn)
        op_layout.addLayout(op_input_layout)

        item_group = QGroupBox("道具列表")
        item_layout = QVBoxLayout(item_group)
        self.rec_item_table = QTableWidget()
        self.rec_item_table.setColumnCount(2)
        self.rec_item_table.setHorizontalHeaderLabels(["道具名", "次数"])
        self.rec_item_table.setMinimumHeight(100)
        self.rec_item_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        item_layout.addWidget(self.rec_item_table)
        item_input_layout = QHBoxLayout()
        self.rec_item_input = QLineEdit()
        self.rec_item_input.setPlaceholderText("道具名...")
        self.rec_item_charges_input = QSpinBox()
        self.rec_item_charges_input.setRange(1, 999)
        self.rec_item_charges_input.setValue(1)
        self.rec_item_add_btn = QPushButton("添加")
        self.rec_item_remove_btn = QPushButton("删除")
        item_input_layout.addWidget(self.rec_item_input)
        item_input_layout.addWidget(self.rec_item_charges_input)
        item_input_layout.addWidget(self.rec_item_add_btn)
        item_input_layout.addWidget(self.rec_item_remove_btn)
        item_layout.addLayout(item_input_layout)

        list_layout = QHBoxLayout()
        list_layout.addWidget(op_group, 1)
        list_layout.addWidget(item_group, 1)
        param_layout.addLayout(list_layout, 3, 0, 1, 4)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 绑定录制器列表按钮
        self.rec_op_add_btn.clicked.connect(self._rec_add_operator)
        self.rec_op_remove_btn.clicked.connect(self._rec_remove_operator)
        self.rec_item_add_btn.clicked.connect(self._rec_add_item)
        self.rec_item_remove_btn.clicked.connect(self._rec_remove_item)

        # 录制参数变更自动保存配置
        self.rec_stage_name.textChanged.connect(self._save_config)
        self.rec_stage_code.textChanged.connect(self._save_config)
        self.rec_grid_rows.valueChanged.connect(self._save_config)
        self.rec_grid_cols.valueChanged.connect(self._save_config)
        self.rec_chk_debug.stateChanged.connect(self._save_config)

        # 使用说明
        guide_label = QLabel()
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.TextFormat.RichText)
        guide_label.setText(
            "<h3>使用说明</h3>"
            "<ul>"
            "<li>请在<span style='color: red;'>全程 1 倍速</span>下使用录制功能，F10停止录制。</li>"
            "<li>请使用<b>快捷键</b>进行暂停，不要鼠标点击暂停按钮。</li>"
            "<li>不要在选中干员的情况下，再选中其他干员进行干员间的跳转。</li>"
            "<li>检测到正式开始游戏(费用条开始动)后自动开始计时。</li>"
            "<li>采用状态机录制有效操作，暂不支持游戏中获得道具，会打乱所有录制，每个人习惯差异很大，以及精度有限，无法保证录制完全准确。</li>"
            "</ul>"
        )
        layout.addWidget(guide_label)

        # 控制按钮
        btn_layout = QHBoxLayout()
        self.btn_rec_start = QPushButton("开始录制")
        self.btn_rec_start.setStyleSheet("background-color: #f44336; color: white;")
        self.btn_rec_start.clicked.connect(self._start_recording)
        btn_layout.addWidget(self.btn_rec_start)

        self.btn_rec_stop = QPushButton("停止录制")
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_stop.clicked.connect(self._stop_recording)
        btn_layout.addWidget(self.btn_rec_stop)

        self.btn_rec_save = QPushButton("导出脚本")
        self.btn_rec_save.setEnabled(False)
        self.btn_rec_save.clicked.connect(self._save_recording)
        btn_layout.addWidget(self.btn_rec_save)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.rec_status = QLabel("状态: 就绪")
        layout.addWidget(self.rec_status)

        layout.addStretch()

    def _parse_recorder_operators(self) -> list:
        return [
            self.rec_op_list.item(i).text()
            for i in range(self.rec_op_list.count())
            if self.rec_op_list.item(i).text().strip()
        ]

    def _parse_recorder_items(self) -> list:
        items = []
        for r in range(self.rec_item_table.rowCount()):
            name_item = self.rec_item_table.item(r, 0)
            charges_item = self.rec_item_table.item(r, 1)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            charges = 1
            if charges_item is not None:
                try:
                    charges = int(charges_item.text())
                except ValueError:
                    charges = 1
            items.append(ItemInfo(name=name, charges=charges))
        return items

    def _start_recording(self):
        try:
            self._recorder_capture = WindowCapture(backend="mss")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"窗口捕获初始化失败:\n{e}")
            return

        operators = self._parse_recorder_operators()
        items = self._parse_recorder_items()
        grid_rows = self.rec_grid_rows.value()
        grid_cols = self.rec_grid_cols.value()
        stage_name = self.rec_stage_name.text().strip() or None
        stage_code = self.rec_stage_code.text().strip() or None

        self._recorder = ActionRecorder(
            capture=self._recorder_capture,
            timer=self._region_timer,
            operators=operators,
            items=items,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            stage_code=stage_code,
            stage_name=stage_name,
            debug=self.rec_chk_debug.isChecked(),
        )
        self._recorder.start()

        # 启动轮询，检测 F10 触发的停止请求及状态更新
        self._recorder_poll_timer = QTimer(self)
        self._recorder_poll_timer.timeout.connect(self._poll_recorder_state)
        self._recorder_poll_timer.start(100)

        self.btn_rec_start.setEnabled(False)
        self.btn_rec_stop.setEnabled(True)
        self.btn_rec_save.setEnabled(False)
        self.rec_status.setText("状态: 等待计时器启动...")

    def _poll_recorder_state(self):
        if self._recorder is None:
            return
        if self._recorder.is_stop_requested():
            self._stop_recording()
            return
        # 若刚从 WAITING_FOR_START 进入 IDLE，更新状态文本
        if self._recorder.is_recording() and hasattr(self._recorder, '_state'):
            if self._recorder._state == "IDLE" and self.rec_status.text().startswith("状态: 等待"):
                self.rec_status.setText("状态: 录制中")

    def _stop_recording(self):
        if hasattr(self, "_recorder_poll_timer") and self._recorder_poll_timer is not None:
            self._recorder_poll_timer.stop()
            self._recorder_poll_timer = None
        if self._recorder is None:
            return
        script = self._recorder.stop()
        self._last_recorded_script = script
        self._recorder = None
        self._recorder_capture = None

        self.btn_rec_start.setEnabled(True)
        self.btn_rec_stop.setEnabled(False)
        self.btn_rec_save.setEnabled(True)
        self.rec_status.setText(f"状态: 录制完成，共 {len(script.actions)} 个操作")

    def _save_recording(self):
        if not hasattr(self, "_last_recorded_script") or self._last_recorded_script is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "保存录制脚本", "", "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_recorded_script.model_dump(), f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "保存成功", f"已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", str(e))

    def _rec_add_operator(self):
        name = self.rec_op_input.text().strip()
        if name:
            self.rec_op_list.addItem(name)
            self.rec_op_input.clear()
            self._save_config()

    def _rec_remove_operator(self):
        for item in self.rec_op_list.selectedItems():
            self.rec_op_list.takeItem(self.rec_op_list.row(item))
        self._save_config()

    def _rec_add_item(self):
        name = self.rec_item_input.text().strip()
        if not name:
            return
        charges = self.rec_item_charges_input.value()
        row = self.rec_item_table.rowCount()
        self.rec_item_table.insertRow(row)
        self.rec_item_table.setItem(row, 0, QTableWidgetItem(name))
        self.rec_item_table.setItem(row, 1, QTableWidgetItem(str(charges)))
        self.rec_item_input.clear()
        self._save_config()

    def _rec_remove_item(self):
        rows = sorted({idx.row() for idx in self.rec_item_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.rec_item_table.removeRow(r)
        self._save_config()

    # ============================================================
    # 使用指南
    # ============================================================
    def _build_guide_tab(self):
        self.guide_tab = QWidget()
        layout = QVBoxLayout(self.guide_tab)

        guide_text = QTextEdit()
        guide_text.setReadOnly(True)
        guide_text.setHtml(
            """
            <h2>Arknights Auto 使用指南</h2>
            <h3>1. 脚本编辑</h3>
            <p>在「脚本编辑」标签页中创建或修改 JSON 脚本。</p>
            <ul>
                <li><b>关卡名</b>：用于 OCR 校验，确保进入了正确的关卡。</li>
                <li><b>关卡代号</b>：用于查询相机位置（levels.json），大多时候与关卡名一致。</li>
                <li><b>地图行列</b>：地图的总行数和总列数，左上为(0,0)，向下加一行，向右加一列，可视化可以查阅各种MAP网站。</li>
                <li><b>干员列表</b>：按<span style="color: red;">进入游戏后初始部署栏顺序从左往右</span>逐个添加。</li>
                <li><b>道具列表</b>：同上添加道具名和可用次数。</li>
            </ul>
            <h3>2. 时间轴操作</h3>
            <p>操作类型包括：deploy（部署）、retreat（撤退）、skill（技能）、
            speed_up（加速）、speed_down（减速）、pause（暂停）、add_item（部署区道具）。</p>
            <p><b>格子格式</b>：行,列（例如 3,2）。</p>
            <p><b>方向</b>：up / down / left / right。</p>
            <p><b>装置</b>：勾选 is_object 表示目标为场上装置/衍生物，不通过部署栏选中。</p>
            <p><b>add_item（部署区新增道具）</b>：当击杀敌人，获得召唤物等获得额外可部署道具时使用。
            格子填写"序号,次数"，其中序号为该道具在道具区域中的从左到右位置（0 表示紧挨着干员的最左侧，数字越大越靠右），次数为可使用次数。</p>
            <p>为保证操作精度，所有操作在子弹时间下进行，<span style="color: red;">请保证部署区始终存在单位，即哪怕单人图，也请携带1名不下场干员</span></p>
            
            <h3>3. 脚本执行</h3>
            <p>在「脚本执行」标签页中选择脚本并运行，运行时不要遮挡游戏屏幕。</p>
            <ul>
                <li><b>无限凸图</b>：脚本结束后自动重新挑战。</li>
                <li><b>漏怪检测</b>：检测到漏怪后自动退出并补打一次（仅一次，不会无限循环）。</li>
                <li><b>Debug</b>：输出调试日志，仅调试BUG使用，平常开启会降低性能。</li>
                <li><b>直接开始作战</b>：跳过 OCR 查找关卡和"开始行动"点击。适用于已手动进入干员编队界面的场景，此时助战参数不可用，请自行选好助战。</li>
                <li><b>突袭模式</b>：进关卡前会选择突袭。</li>
                <li><b>助战参数</b>：不借用则不勾选，好友位从左到右依次为0到9，点击助战后不做任何移动，技能从左到右依此为1到3，模组从左到右依此为1到3。</li>
                <li><b>键位设置</b>：<span style="color: red;">键位务必和游戏中对应快捷键位一致</span>，可以下拉选择，也可以输入。运行一次脚本后自动保存。</li>
            </ul>
            <h3>4. 资源更新</h3>
            <p>在「资源更新」标签页中可以上传新的 levels.json 文件。</p>
            levels.json为游戏解包关卡资源，包含了不同关卡相机位置，为了精准对齐格子需要更新加载。<br>
            资源可在https://github.com/yuanyan3060/ArknightsGameResource中获取，请为该解包和格子对齐项目点上star吧！<br>
            <h3>5. 快捷键</h3>
            <ul>
                <li><b>F11</b>：暂停/恢复脚本</li>
                <li><b>F12</b>：紧急暂停脚本并暂停游戏</li>
            </ul>
            <h3>6. 其他注意事项</h3>
            <ul>
                <li><b>UI设置</b>：UI设置请采用默认的90大小，否则脚本无法有效执行。</li>
                <li><b>管理员权限</b>：请以管理员模式启动，否则游戏无法接受键位和鼠标操作。</li>
            <ul>
            """
        )
        layout.addWidget(guide_text)
