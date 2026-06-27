from __future__ import annotations

import json
import os
import sys
from typing import TYPE_CHECKING

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
from gui.tabs import (
    ExecTab,
    EditorTab,
    ResourceTab,
    MatchstickTab,
    TimerTab,
    RecorderTab,
    GuideTab,
)

if TYPE_CHECKING:
    from gui.tabs.exec_tab import ExecTab
    from gui.tabs.editor_tab import EditorTab
    from gui.tabs.resource_tab import ResourceTab
    from gui.tabs.matchstick_tab import MatchstickTab
    from gui.tabs.timer_tab import TimerTab
    from gui.tabs.recorder_tab import RecorderTab
    from gui.tabs.guide_tab import GuideTab


class MainWindow(QMainWindow):
    # --- 跨 Tab 共享的 UI 控件类型标注（由 gui/tabs 下各 Tab 在构建时赋值） ---
    exec_tab: ExecTab
    editor_tab: EditorTab
    resource_tab: ResourceTab
    matchstick_tab: MatchstickTab
    timer_tab: TimerTab
    recorder_tab: RecorderTab
    guide_tab: GuideTab

    # 脚本执行 Tab
    exec_script_path: QLineEdit
    btn_browse: QPushButton
    chk_loop: QCheckBox
    chk_leak: QCheckBox
    chk_debug: QCheckBox
    chk_direct_start: QCheckBox
    chk_challenge_mode: QCheckBox
    chk_borrow_support: QCheckBox
    spin_support_friend: QSpinBox
    combo_support_skill: QComboBox
    combo_support_module: QComboBox
    combo_pause_key: QComboBox
    line_skill_key: QLineEdit
    line_retreat_key: QLineEdit
    btn_run: QPushButton
    btn_stop: QPushButton
    status_label: QLabel
    log_text: QTextEdit
    process: QProcess | None

    # 脚本编辑 Tab
    stage_name_edit: QLineEdit
    stage_code_edit: QLineEdit
    rows_spin: QSpinBox
    cols_spin: QSpinBox
    operators_list: QListWidget
    op_input: QLineEdit
    btn_add_op: QPushButton
    btn_remove_op: QPushButton
    btn_up_op: QPushButton
    btn_down_op: QPushButton
    items_table: QTableWidget
    item_input: QLineEdit
    item_charges_input: QSpinBox
    btn_add_item: QPushButton
    btn_remove_item: QPushButton
    btn_up_item: QPushButton
    btn_down_item: QPushButton
    summons_table: QTableWidget
    summon_input: QLineEdit
    summon_cost_input: QSpinBox
    btn_add_summon: QPushButton
    btn_remove_summon: QPushButton
    btn_up_summon: QPushButton
    btn_down_summon: QPushButton
    action_table: QTableWidget
    btn_add: QPushButton
    btn_remove: QPushButton
    btn_up: QPushButton
    btn_down: QPushButton
    edit_time_s: QSpinBox
    edit_time_f: QSpinBox
    _action_labels: dict
    _action_labels_rev: dict
    combo_action: QComboBox
    combo_op: QComboBox
    grid_input_widget: QWidget
    edit_grid: QLineEdit
    item_index_widget: QWidget
    edit_item_index: QSpinBox
    edit_item_charges: QSpinBox
    edit_dir: QComboBox
    chk_is_object: QCheckBox
    btn_new: QPushButton
    btn_open: QPushButton
    btn_save: QPushButton

    # 资源更新 Tab
    resource_path: QLineEdit
    btn_resource_browse: QPushButton
    btn_update_resource: QPushButton
    resource_status: QLabel

    # 划火柴 Tab
    chk_matchstick_select: QCheckBox
    line_matchstick_select: QLineEdit
    chk_matchstick_166: QCheckBox
    line_matchstick_166: QLineEdit
    chk_matchstick_50: QCheckBox
    line_matchstick_50: QLineEdit

    # 计时器 Tab
    btn_timer_start: QPushButton
    btn_timer_stop: QPushButton
    btn_timer_reset: QPushButton
    chk_timer_debug: QCheckBox
    timer_status: QLabel

    # 操作录制 Tab
    rec_stage_name: QLineEdit
    rec_stage_code: QLineEdit
    rec_grid_rows: QSpinBox
    rec_grid_cols: QSpinBox
    rec_chk_debug: QCheckBox
    rec_op_list: QListWidget
    rec_op_input: QLineEdit
    rec_op_add_btn: QPushButton
    rec_op_remove_btn: QPushButton
    rec_item_table: QTableWidget
    rec_item_input: QLineEdit
    rec_item_charges_input: QSpinBox
    rec_item_add_btn: QPushButton
    rec_item_remove_btn: QPushButton
    btn_rec_start: QPushButton
    btn_rec_stop: QPushButton
    btn_rec_save: QPushButton
    rec_status: QLabel
    _recorder_poll_timer: QTimer | None
    _last_recorded_script: ScriptModel | None

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
            "cost_tag": self.combo_cost_tag.currentData() or "",
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
        cost_tag = config.get("cost_tag", "")
        if cost_tag:
            idx = self.combo_cost_tag.findData(cost_tag)
            if idx >= 0:
                self.combo_cost_tag.setCurrentIndex(idx)
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

        self.exec_tab._on_loop_changed(self.chk_loop.checkState())
        self.exec_tab._on_direct_start_changed(self.chk_direct_start.checkState())
        self.exec_tab._on_challenge_mode_changed(self.chk_challenge_mode.checkState())
        self.exec_tab._on_borrow_support_changed(self.chk_borrow_support.checkState())
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
            matchstick_hotkeys = self.timer_tab._build_matchstick_hotkeys()
            self._region_timer.update_matchstick_hotkeys(
                matchstick_hotkeys if matchstick_hotkeys else None
            )
        self._save_config()

    def _build_ui(self):
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        self.exec_tab = ExecTab(self)
        tabs.addTab(self.exec_tab, "脚本执行")

        self.editor_tab = EditorTab(self)
        tabs.addTab(self.editor_tab, "脚本编辑")

        self.resource_tab = ResourceTab(self)
        tabs.addTab(self.resource_tab, "资源更新")

        self.matchstick_tab = MatchstickTab(self)
        tabs.addTab(self.matchstick_tab, "划火柴")

        self.timer_tab = TimerTab(self)
        tabs.addTab(self.timer_tab, "计时器")

        self.recorder_tab = RecorderTab(self)
        tabs.addTab(self.recorder_tab, "操作录制")

        self.guide_tab = GuideTab(self)
        tabs.addTab(self.guide_tab, "使用指南")

        # 所有 UI 控件创建完成后再加载配置，避免信号处理时访问未创建的控件
        self._apply_config(self._load_config())

    # 以下方法仍由外部/Tab 通过 main_window 调用，保持公共接口
    def _on_matchstick_enabled_changed(self, state):
        self._apply_matchstick_config()

    def _on_matchstick_hotkey_changed(self, text):
        sender = self.sender()
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
            sender.blockSignals(True)
            cfg = action.get_matchstick_config()
            sender.setText(cfg["hotkeys"].get(self.exec_tab._matchstick_op_from_widget(sender), ""))
            sender.blockSignals(False)
            return
        self._apply_matchstick_config()

    def _stop_region_timer(self):
        self.timer_tab._stop_region_timer()

    # 保留 process 相关公共状态访问，但方法已移至 ExecTab
    # 保留 _region_timer 等公共状态
