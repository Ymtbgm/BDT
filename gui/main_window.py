from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox,
    QFileDialog, QSpinBox, QComboBox, QTabWidget,
    QCheckBox, QTextEdit, QGroupBox, QListWidget,
    QSizePolicy, QProgressBar,
)
from PyQt6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer
from PyQt6.QtGui import QIcon, QPixmap, QPainter

import action
from core.capture.capture import WindowCapture
from core.game_state.region_state_timer import RegionStateTimer
from core.recording.recorder import ActionRecorder
from core.base.paths import get_project_root, gui_template
from gui.timer_overlay import TimerOverlay
from gui.widgets.checked_combo_box import CheckedComboBox
from models.script_schema import ScriptModel, OperatorAction, ActionType, ItemInfo
from gui.tabs import (
    ExecTab,
    EditorTab,
    ResourceTab,
    MatchstickTab,
    TimerTab,
    RecorderTab,
)

if TYPE_CHECKING:
    from gui.tabs.exec_tab import ExecTab
    from gui.tabs.editor_tab import EditorTab
    from gui.tabs.resource_tab import ResourceTab
    from gui.tabs.matchstick_tab import MatchstickTab
    from gui.tabs.timer_tab import TimerTab
    from gui.tabs.recorder_tab import RecorderTab


class MainWindow(QMainWindow):
    # --- 跨 Tab 共享的 UI 控件类型标注（由 gui/tabs 下各 Tab 在构建时赋值） ---
    exec_tab: ExecTab
    editor_tab: EditorTab
    resource_tab: ResourceTab
    matchstick_tab: MatchstickTab
    timer_tab: TimerTab
    recorder_tab: RecorderTab

    # 脚本执行 Tab
    exec_script_path: QLineEdit
    btn_browse: QPushButton
    chk_loop: QCheckBox
    chk_leak: QCheckBox
    chk_debug: QCheckBox
    chk_direct_start: QCheckBox
    chk_challenge_mode: QCheckBox
    chk_speed2x: QCheckBox
    chk_borrow_support: QCheckBox
    btn_support_config: QPushButton
    spin_support_friend: QSpinBox
    combo_support_skill: QComboBox
    combo_support_module: QComboBox
    combo_pause_key: QComboBox
    line_skill_key: QLineEdit
    line_retreat_key: QLineEdit
    line_speed_key: QLineEdit
    btn_run: QPushButton
    btn_stop: QPushButton
    status_label: QLabel
    log_text: QTextEdit
    process: QProcess | None

    # 脚本编辑 Tab
    stage_code_edit: QLineEdit
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
    chk_levels_auto_update: QCheckBox
    chk_levels_auto_download: QCheckBox
    line_levels_metadata_url: QLineEdit
    btn_levels_check_update: QPushButton
    progress_levels_update: QProgressBar
    label_levels_update_status: QLabel

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
    rec_stage_code: QLineEdit
    rec_loaded_script_path: QLineEdit
    rec_loaded_script_status: QLabel
    btn_rec_load_script: QPushButton
    btn_rec_clear_script: QPushButton
    rec_chk_support_op: QCheckBox
    rec_takeover_hotkey: QLineEdit
    rec_chk_probability_retry: QCheckBox
    btn_rec_probability_config: QPushButton
    combo_rec_debug: CheckedComboBox
    rec_initial_operator_count: QSpinBox
    rec_initial_item_count: QSpinBox
    rec_cost_tag: QComboBox
    rec_op_table: QTableWidget
    rec_op_input: QLineEdit
    rec_op_cost_input: QSpinBox
    rec_op_add_btn: QPushButton
    rec_op_remove_btn: QPushButton
    rec_op_up_btn: QPushButton
    rec_op_down_btn: QPushButton
    rec_item_table: QTableWidget
    rec_item_input: QLineEdit
    rec_item_charges_input: QSpinBox
    rec_item_add_btn: QPushButton
    rec_item_remove_btn: QPushButton
    rec_item_up_btn: QPushButton
    rec_item_down_btn: QPushButton
    btn_rec_start: QPushButton
    btn_rec_stop: QPushButton
    btn_rec_save: QPushButton
    rec_status: QLabel
    _recorder_poll_timer: QTimer | None
    _last_recorded_script: ScriptModel | None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blood Demon Toolbox")
        # 兼容开发环境与 PyInstaller onedir 打包后的图标路径
        _icon_path = str(gui_template("Icon.ico"))
        if not os.path.exists(_icon_path):
            # 兜底：旧位置或当前工作目录
            _icon_candidates = [
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "core", "resource", "Icon.ico"),
                os.path.join(os.getcwd(), "resource", "gui_template", "Icon.ico"),
                os.path.join(os.getcwd(), "core", "resource", "Icon.ico"),
            ]
            _icon_path = next((p for p in _icon_candidates if os.path.exists(p)), _icon_path)
        self.setWindowIcon(QIcon(_icon_path))
        self.resize(1080, 608)
        self.setMinimumWidth(1020)
        self.script = ScriptModel(stage_code="1-7")
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
        """退出时停止划火柴全局热键监听、区域计时器，并关闭所有悬浮窗。"""
        action.stop_matchstick_listener()
        self._stop_region_timer()
        # 关闭录制器悬浮窗，避免主程序退出后残留独立窗口
        recorder_overlay = getattr(self, "_recorder_overlay", None)
        if recorder_overlay is not None:
            try:
                recorder_overlay.close()
            except Exception:
                pass
            self._recorder_overlay = None
        event.accept()

    def _project_root(self) -> str:
        return str(get_project_root())

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
        """计时器悬浮窗专用：使用 floor 取帧，避免初始/边界帧数多 1。

        若当前存在费用条校准表，会按校准表的 frame_duration_ms 转换当前秒内
        的帧号，以适配普通模式 10s 前后、危机合约 tag 等不同循环。
        """
        frame_duration = 33.333
        cycle_length = 30
        region_timer = getattr(self, "_region_timer", None)
        if region_timer is not None:
            cost_sync = getattr(region_timer, "cost_sync", None)
            if cost_sync is not None and hasattr(cost_sync, "get_calibration"):
                try:
                    cal = cost_sync.get_calibration(ms)
                    if cal is not None and cal.frame_duration_ms and cal.cycle_length:
                        frame_duration = cal.frame_duration_ms
                        cycle_length = cal.cycle_length
                except Exception:
                    pass
            # 费用不自然回复模式下费用条同步被禁用，使用默认 30fps

        s = int(ms) // 1000
        # 加极小 epsilon，避免费用条帧边界处的浮点误差把 frame 29 显示成 28
        f = int((ms % 1000 + 1e-5) / frame_duration)
        f = min(cycle_length - 1, max(0, f))
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
            self._normalize_key_name(self.line_speed_key.text()),
        }

    def _save_config(self):
        path = self._config_path()
        # 先读取已有配置，保留本方法未显式维护的键（如 ui_scale_check_disabled）
        config = self._load_config()
        config.update({
            "last_script_path": self.exec_script_path.text(),
            "loop": self.chk_loop.isChecked(),
            "leak": self.chk_leak.isChecked(),
            "debug": self.chk_debug.isChecked(),
            "direct_start": self.chk_direct_start.isChecked(),
            "challenge_mode": self.chk_challenge_mode.isChecked(),
            "sand_table": self.chk_sand_table.isChecked(),
            "speed2x": self.chk_speed2x.isChecked(),
            "cost_tag": self.combo_cost_tag.currentData() or "",
            "rec_debug_keys": self.combo_rec_debug.checked_data(),
            "rec_support_op": self.rec_chk_support_op.isChecked(),
            "rec_takeover_hotkey": self.rec_takeover_hotkey.text().strip().upper() or "F9",
            "rec_probability_retry_enabled": self.rec_chk_probability_retry.isChecked(),
            "rec_probability_retry_config": getattr(
                self.recorder_tab, "_probability_retry_config",
                {
                    "challenge_mode": False,
                    "sand_table": False,
                    "borrow_support": False,
                    "support_friend_index": 0,
                    "support_skill": 1,
                    "support_module": 1,
                }
            ),
            "rec_stage_code": self.rec_stage_code.text(),
            "rec_loaded_script_path": self.rec_loaded_script_path.text(),
            "rec_cost_tag": self.rec_cost_tag.currentData() or "",
            "rec_initial_operator_count": self.rec_initial_operator_count.value(),
            "rec_initial_item_count": self.rec_initial_item_count.value(),
            "rec_operators": [
                {
                    "name": self.rec_op_table.item(r, 0).text(),
                    "cost": int(self.rec_op_table.item(r, 1).text()) if self.rec_op_table.item(r, 1) and self.rec_op_table.item(r, 1).text().isdigit() else None,
                }
                for r in range(self.rec_op_table.rowCount())
            ],
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
            "speed_key": self._normalize_key_name(self.line_speed_key.text()),
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
            "levels_auto_update": {
                "enabled": self.chk_levels_auto_update.isChecked(),
                "auto_download": self.chk_levels_auto_download.isChecked(),
                "metadata_url": self.line_levels_metadata_url.text(),
            },
        })
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
        challenge_mode = config.get("challenge_mode", False)
        sand_table = config.get("sand_table", False)
        # 突袭与沙盘推演互斥，同时开启时优先关闭沙盘推演
        if challenge_mode and sand_table:
            sand_table = False
        self.chk_challenge_mode.setChecked(challenge_mode)
        self.chk_sand_table.setChecked(sand_table)
        self.chk_speed2x.setChecked(config.get("speed2x", False))
        cost_tag = config.get("cost_tag", "")
        if cost_tag:
            idx = self.combo_cost_tag.findData(cost_tag)
            if idx >= 0:
                self.combo_cost_tag.setCurrentIndex(idx)
        self.combo_rec_debug.set_checked_data(config.get("rec_debug_keys", []))
        # 兼容旧配置：根据原来的布尔值构造勾选列表
        legacy_keys = []
        if config.get("rec_debug"):
            legacy_keys.append("recorder")
        if config.get("rec_debug_cost_bar"):
            legacy_keys.append("cost_bar")
        if config.get("rec_debug_resolver"):
            legacy_keys.append("resolver")
        if config.get("rec_debug_screenshot"):
            legacy_keys.append("screenshot")
        if legacy_keys:
            self.combo_rec_debug.set_checked_data(legacy_keys)
        self.rec_chk_support_op.setChecked(config.get("rec_support_op", False))
        self.rec_takeover_hotkey.setText(config.get("rec_takeover_hotkey", "F9"))
        self.rec_chk_probability_retry.setChecked(config.get("rec_probability_retry_enabled", False))
        retry_cfg = config.get("rec_probability_retry_config", {})
        if isinstance(retry_cfg, dict):
            self.recorder_tab._probability_retry_config = retry_cfg
        self.rec_stage_code.setText(config.get("rec_stage_code", ""))
        self.rec_loaded_script_path.setText(config.get("rec_loaded_script_path", ""))
        rec_cost_tag = config.get("rec_cost_tag", "")
        if rec_cost_tag:
            idx = self.rec_cost_tag.findData(rec_cost_tag)
            if idx >= 0:
                self.rec_cost_tag.setCurrentIndex(idx)
        self.rec_initial_operator_count.setValue(config.get("rec_initial_operator_count", 12))
        self.rec_initial_item_count.setValue(config.get("rec_initial_item_count", 0))
        self.rec_op_table.setRowCount(0)
        for op in config.get("rec_operators", []):
            name = ""
            cost = None
            if isinstance(op, dict):
                name = op.get("name", "")
                cost = op.get("cost")
            elif isinstance(op, str):
                name = op
            if not name:
                continue
            row = self.rec_op_table.rowCount()
            self.rec_op_table.insertRow(row)
            self.rec_op_table.setItem(row, 0, QTableWidgetItem(name))
            cost_text = str(cost) if isinstance(cost, int) and cost > 0 else ""
            self.rec_op_table.setItem(row, 1, QTableWidgetItem(cost_text))
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

        pause_key = self._normalize_key_name(config.get("pause_key", "space"))
        idx = self.combo_pause_key.findText(pause_key)
        if idx >= 0:
            self.combo_pause_key.setCurrentIndex(idx)
        else:
            self.combo_pause_key.setCurrentText(pause_key)

        self.line_skill_key.setText(self._normalize_key_name(config.get("skill_key", "e")))
        self.line_retreat_key.setText(self._normalize_key_name(config.get("retreat_key", "q")))
        self.line_speed_key.setText(self._normalize_key_name(config.get("speed_key", "f")))

        # 同步游戏内快捷键到 action 模块（GUI 进程也运行划火柴监听，需要一致）
        action.configure_keys(
            pause=self._normalize_key_name(config.get("pause_key", "space")),
            skill=self._normalize_key_name(config.get("skill_key", "e")),
            retreat=self._normalize_key_name(config.get("retreat_key", "q")),
            speed=self._normalize_key_name(config.get("speed_key", "f")),
        )

        # levels.json 自动更新配置
        levels_auto_update = config.get("levels_auto_update", {})
        self.chk_levels_auto_update.setChecked(levels_auto_update.get("enabled", True))
        self.chk_levels_auto_download.setChecked(levels_auto_update.get("auto_download", True))
        self.line_levels_metadata_url.setText(
            levels_auto_update.get("metadata_url", "")
        )
        # 启用状态联动
        self.chk_levels_auto_download.setEnabled(self.chk_levels_auto_update.isChecked())

        # 划火柴热键配置
        matchstick = config.get("matchstick", {})
        for op, widget_key, widget_chk in (
            ("select_operator", self.line_matchstick_select, self.chk_matchstick_select),
            ("pass_166ms", self.line_matchstick_166, self.chk_matchstick_166),
            ("pass_50ms", self.line_matchstick_50, self.chk_matchstick_50),
        ):
            op_cfg = matchstick.get(op, {})
            widget_key.setText(self._normalize_hotkey(op_cfg.get("hotkey", {"select_operator": "r", "pass_166ms": "p", "pass_50ms": "f"}[op])))
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

    def _tab_icon(self, filename: str, size: int = 25, offset_y: int = 0) -> QIcon:
        """加载并缩放 ui_icons 目录下的图标，统一成相同大小；offset_y 可向下微调。"""
        icon_path = str(gui_template("ui_icons") / filename)
        pixmap = QPixmap(icon_path)
        if pixmap.isNull():
            return QIcon()
        scaled = pixmap.scaled(
            size, size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if offset_y == 0:
            return QIcon(scaled)
        # 整体画布略大，把图标向下偏移 offset_y 像素
        target = QPixmap(size, size + offset_y)
        target.fill(Qt.GlobalColor.transparent)
        painter = QPainter(target)
        painter.drawPixmap(0, offset_y, scaled)
        painter.end()
        return QIcon(target)

    def _build_ui(self):
        tabs = QTabWidget()
        tabs.setStyleSheet("QTabBar::tab { font-weight: bold; }")
        self.setCentralWidget(tabs)

        self.exec_tab = ExecTab(self)
        tabs.addTab(self.exec_tab, "脚本执行")
        tabs.setTabIcon(tabs.indexOf(self.exec_tab), self._tab_icon("execute.png"))

        self.editor_tab = EditorTab(self)
        tabs.addTab(self.editor_tab, "脚本编辑")
        tabs.setTabIcon(tabs.indexOf(self.editor_tab), self._tab_icon("edit.png", offset_y=3))

        self.resource_tab = ResourceTab(self)
        tabs.addTab(self.resource_tab, "资源更新")
        tabs.setTabIcon(tabs.indexOf(self.resource_tab), self._tab_icon("update_file.png"))

        self.matchstick_tab = MatchstickTab(self)
        tabs.addTab(self.matchstick_tab, "划火柴")
        tabs.setTabIcon(tabs.indexOf(self.matchstick_tab), self._tab_icon("match.png"))

        self.timer_tab = TimerTab(self)
        tabs.addTab(self.timer_tab, "计时器")
        tabs.setTabIcon(tabs.indexOf(self.timer_tab), self._tab_icon("timer.png", offset_y=3))

        self.recorder_tab = RecorderTab(self)
        tabs.addTab(self.recorder_tab, "操作录制")
        tabs.setTabIcon(tabs.indexOf(self.recorder_tab), self._tab_icon("record.png"))

        # 所有 UI 控件创建完成后再加载配置，避免信号处理时访问未创建的控件
        self._apply_config(self._load_config())

        # 若启用自动检查，启动后后台静默检查更新
        self.resource_tab.trigger_auto_check()

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
