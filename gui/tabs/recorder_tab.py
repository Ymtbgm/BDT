import json
import time
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox,
    QGroupBox, QListWidget, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QFileDialog, QComboBox,
    QApplication, QInputDialog, QDialog, QDialogButtonBox,
)
from PyQt6.QtCore import Qt, QTimer, QMetaObject, Q_ARG, QThread

import action
from core.base.paths import get_project_root
from gui.tabs._ui_utils import set_group_box_icon
from core.capture.capture import WindowCapture
from core.vision.ocr_engine import OCREngine
from core.recording.recorder import ActionRecorder
from core.game_state.region_state_timer import RegionStateTimer
from core.map.tile_pos import load_stage_dimensions
from gui._window_effects import (
    remove_dwm_glass_border,
    set_window_topmost,
    set_tool_window_style,
)
from gui.info_collection_overlay import InfoCollectionOverlay
from gui.widgets.checked_combo_box import CheckedComboBox
from models.script_schema import ItemInfo, ScriptModel


class ProbabilityRetryConfigDialog(QDialog):
    """录制器装载脚本时的概率点自动凸图配置对话框。"""

    def __init__(self, parent=None, config=None):
        super().__init__(parent)
        self.setWindowTitle("概率点配置")
        self.setMinimumWidth(300)
        self._config = config or {}
        self._build_ui()
        self._load_config()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        mode_layout = QHBoxLayout()
        self.chk_challenge_mode = QCheckBox("突袭模式")
        self.chk_sand_table = QCheckBox("沙盘推演")
        self.chk_challenge_mode.stateChanged.connect(self._on_challenge_changed)
        self.chk_sand_table.stateChanged.connect(self._on_sand_table_changed)
        mode_layout.addWidget(self.chk_challenge_mode)
        mode_layout.addWidget(self.chk_sand_table)
        mode_layout.addStretch()
        layout.addLayout(mode_layout)

        support_group = QGroupBox("助战参数")
        support_layout = QHBoxLayout(support_group)
        self.chk_borrow_support = QCheckBox("借用助战干员")
        self.chk_borrow_support.stateChanged.connect(self._on_borrow_support_changed)
        support_layout.addWidget(self.chk_borrow_support)

        support_layout.addWidget(QLabel("好友位置"))
        self.spin_friend_index = QSpinBox()
        self.spin_friend_index.setRange(0, 8)
        support_layout.addWidget(self.spin_friend_index)

        support_layout.addWidget(QLabel("携带技能"))
        self.combo_skill = QComboBox()
        self.combo_skill.addItems(["1", "2", "3"])
        support_layout.addWidget(self.combo_skill)

        support_layout.addWidget(QLabel("模组选择"))
        self.combo_module = QComboBox()
        self.combo_module.addItems(["1", "2", "3"])
        support_layout.addWidget(self.combo_module)
        support_layout.addStretch()
        layout.addWidget(support_group)

        hint = QLabel(
            "此处配置关卡重开信息，概率点信息请在脚本编辑中打开脚本，添加“概率点检查”完成配置。"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _load_config(self):
        challenge_mode = self._config.get("challenge_mode", False)
        sand_table = self._config.get("sand_table", False)
        # 突袭与沙盘互斥，同时开启时优先关闭沙盘推演
        if challenge_mode and sand_table:
            sand_table = False
        self.chk_challenge_mode.setChecked(challenge_mode)
        self.chk_sand_table.setChecked(sand_table)

        borrow_support = self._config.get("borrow_support", False)
        self.chk_borrow_support.setChecked(borrow_support)
        self.spin_friend_index.setValue(self._config.get("support_friend_index", 0))
        skill_idx = max(0, min(2, self._config.get("support_skill", 1) - 1))
        self.combo_skill.setCurrentIndex(skill_idx)
        module_idx = max(0, min(2, self._config.get("support_module", 1) - 1))
        self.combo_module.setCurrentIndex(module_idx)
        self._on_borrow_support_changed(self.chk_borrow_support.checkState())

    def _on_challenge_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.chk_sand_table.setChecked(False)

    def _on_sand_table_changed(self, state):
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        if checked:
            self.chk_challenge_mode.setChecked(False)

    def _on_borrow_support_changed(self, state):
        if isinstance(state, Qt.CheckState):
            enabled = state == Qt.CheckState.Checked
        else:
            enabled = state == Qt.CheckState.Checked.value
        self.spin_friend_index.setEnabled(enabled)
        self.combo_skill.setEnabled(enabled)
        self.combo_module.setEnabled(enabled)

    def get_config(self):
        return {
            "challenge_mode": self.chk_challenge_mode.isChecked(),
            "sand_table": self.chk_sand_table.isChecked(),
            "borrow_support": self.chk_borrow_support.isChecked(),
            "support_friend_index": self.spin_friend_index.value(),
            "support_skill": int(self.combo_skill.currentText()),
            "support_module": int(self.combo_module.currentText()),
        }


class RecorderTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 参数区域：每行一个 QHBoxLayout，输入框紧跟标签，不居中展开
        param_group = QGroupBox("录制参数")
        set_group_box_icon(param_group, "level_setup.png")
        param_layout = QVBoxLayout(param_group)
        param_layout.setContentsMargins(6, 6, 6, 6)
        param_layout.setSpacing(6)

        row = QHBoxLayout()
        row.addWidget(QLabel("关卡代号:"))
        self.main_window.rec_stage_code = QLineEdit()
        self.main_window.rec_stage_code.setPlaceholderText("如 1-7")
        self.main_window.rec_stage_code.setMaximumWidth(80)
        row.addWidget(self.main_window.rec_stage_code)
        row.addStretch()
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("装载脚本:"))
        self.main_window.rec_loaded_script_path = QLineEdit()
        self.main_window.rec_loaded_script_path.setPlaceholderText("选择要预执行的脚本...")
        self.main_window.rec_loaded_script_path.setReadOnly(True)
        row.addWidget(self.main_window.rec_loaded_script_path)
        self.main_window.btn_rec_load_script = QPushButton("浏览")
        self.main_window.btn_rec_load_script.clicked.connect(self._browse_loaded_script)
        row.addWidget(self.main_window.btn_rec_load_script)
        self.main_window.btn_rec_clear_script = QPushButton("清除")
        self.main_window.btn_rec_clear_script.clicked.connect(self._clear_loaded_script)
        row.addWidget(self.main_window.btn_rec_clear_script)
        self.main_window.rec_loaded_script_status = QLabel("当前未装载脚本")
        row.addWidget(self.main_window.rec_loaded_script_status)
        row.addStretch()
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("初始干员数量:"))
        self.main_window.rec_initial_operator_count = QSpinBox()
        self.main_window.rec_initial_operator_count.setRange(0, 13)
        self.main_window.rec_initial_operator_count.setValue(12)
        self.main_window.rec_initial_operator_count.setToolTip(
            "编队界面实际携带的干员总数（含助战）。不勾选助战时，右上角助战位会被跳过。"
        )
        row.addWidget(self.main_window.rec_initial_operator_count)
        self.main_window.rec_chk_support_op = QCheckBox("借用助战干员")
        self.main_window.rec_chk_support_op.setToolTip(
            "编队界面第 13 个槽位（右上角）为助战干员；不勾选时该位置会被跳过。"
        )
        row.addWidget(self.main_window.rec_chk_support_op)
        row.addStretch()
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("初始道具数量:"))
        self.main_window.rec_initial_item_count = QSpinBox()
        self.main_window.rec_initial_item_count.setRange(0, 12)
        self.main_window.rec_initial_item_count.setValue(0)
        self.main_window.rec_initial_item_count.setToolTip(
            "部署区初始道具种类数。道具将排列在部署栏最右侧。"
        )
        row.addWidget(self.main_window.rec_initial_item_count)
        row.addStretch()
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("接管快捷键:"))
        self.main_window.rec_takeover_hotkey = QLineEdit()
        self.main_window.rec_takeover_hotkey.setPlaceholderText("F9")
        self.main_window.rec_takeover_hotkey.setMaximumWidth(80)
        self.main_window.rec_takeover_hotkey.setToolTip(
            "装载脚本执行期间，按下该快捷键可立即手动接管，进入用户录制模式。"
        )
        row.addWidget(self.main_window.rec_takeover_hotkey)

        self.main_window.rec_chk_probability_retry = QCheckBox("概率点配置")
        self.main_window.rec_chk_probability_retry.setToolTip(
            "装载脚本执行期间，若概率点检查失败则自动退出并重新进入关卡，直到所有概率点通过。"
        )
        self.main_window.rec_chk_probability_retry.stateChanged.connect(self._on_probability_retry_changed)
        row.addWidget(self.main_window.rec_chk_probability_retry)

        self.main_window.btn_rec_probability_config = QPushButton("配置")
        self.main_window.btn_rec_probability_config.setEnabled(False)
        self.main_window.btn_rec_probability_config.setToolTip(
            "配置概率点重试时的突袭/沙盘/助战参数。"
        )
        self.main_window.btn_rec_probability_config.clicked.connect(self._open_probability_config)
        row.addWidget(self.main_window.btn_rec_probability_config)
        row.addStretch()
        param_layout.addLayout(row)

        # 概率点重试配置缓存（不写入脚本，仅作为运行时选项）
        self._probability_retry_config = {
            "challenge_mode": False,
            "sand_table": False,
            "borrow_support": False,
            "support_friend_index": 0,
            "support_skill": 1,
            "support_module": 1,
        }

        row = QHBoxLayout()
        row.addWidget(QLabel("费用条 tag:"))
        self.main_window.rec_cost_tag = QComboBox()
        self.main_window.rec_cost_tag.addItem("无", "")
        self.main_window.rec_cost_tag.addItem("费用回复降低25%", "cc_25")
        self.main_window.rec_cost_tag.addItem("费用回复降低50%", "cc_50")
        self.main_window.rec_cost_tag.addItem("费用回复降低75%", "cc_75")
        self.main_window.rec_cost_tag.addItem("费用不自然回复", "no_regen")
        self.main_window.rec_cost_tag.setToolTip(
            "选择关卡的费用条校准 tag；通常只有危机合约/特殊关卡需要选择。"
        )
        self.main_window.rec_cost_tag.setMaximumWidth(200)
        row.addWidget(self.main_window.rec_cost_tag)
        row.addStretch()
        param_layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Debug选项:"))
        self.main_window.combo_rec_debug = CheckedComboBox("未选择")
        self.main_window.combo_rec_debug.add_item("录制器状态机", "recorder")
        self.main_window.combo_rec_debug.add_item("费用条检测", "cost_bar")
        self.main_window.combo_rec_debug.add_item("离线识别", "resolver")
        self.main_window.combo_rec_debug.add_item("调试截图", "screenshot")
        self.main_window.combo_rec_debug.add_item("脚本装载执行", "loaded_script")
        self.main_window.combo_rec_debug.add_item("技能状态检测", "skill_status")
        self.main_window.combo_rec_debug.setToolTip(
            "勾选需要打印日志或保存截图的调试项，可多选。"
        )
        self.main_window.combo_rec_debug.setMaximumWidth(200)
        row.addWidget(self.main_window.combo_rec_debug)
        row.addStretch()
        param_layout.addLayout(row)

        # 干员列表 + 道具列表并排
        op_group = QGroupBox("干员列表")
        set_group_box_icon(op_group, "operator.png")
        op_layout = QVBoxLayout(op_group)
        op_layout.setContentsMargins(6, 6, 6, 6)
        self.main_window.rec_op_table = QTableWidget()
        self.main_window.rec_op_table.setColumnCount(2)
        self.main_window.rec_op_table.setHorizontalHeaderLabels(["干员名", "费用"])
        self.main_window.rec_op_table.setMinimumHeight(200)
        self.main_window.rec_op_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.main_window.rec_op_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.DoubleClicked
        )
        op_layout.addWidget(self.main_window.rec_op_table)
        op_input_layout = QHBoxLayout()
        op_input_layout = QHBoxLayout()
        op_input_layout.setSpacing(4)
        self.main_window.rec_op_input = QLineEdit()
        self.main_window.rec_op_input.setPlaceholderText("输入干员名...")
        self.main_window.rec_op_input.setMinimumWidth(100)
        self.main_window.rec_op_cost_input = QSpinBox()
        self.main_window.rec_op_cost_input.setRange(0, 999)
        self.main_window.rec_op_cost_input.setValue(0)
        self.main_window.rec_op_cost_input.setSpecialValueText("未设置")
        self.main_window.rec_op_cost_input.setFixedWidth(95)
        self.main_window.rec_op_add_btn = QPushButton("添加")
        self.main_window.rec_op_remove_btn = QPushButton("删除")
        self.main_window.rec_op_up_btn = QPushButton("上移")
        self.main_window.rec_op_down_btn = QPushButton("下移")
        for btn in (
            self.main_window.rec_op_add_btn,
            self.main_window.rec_op_remove_btn,
            self.main_window.rec_op_up_btn,
            self.main_window.rec_op_down_btn,
        ):
            btn.setFixedHeight(22)
            btn.setFixedWidth(45)
            btn.setStyleSheet("padding: 1px 4px; font-size: 11px;")
        op_input_layout.addWidget(self.main_window.rec_op_input, 1)
        op_input_layout.addWidget(self.main_window.rec_op_cost_input, 0)
        op_input_layout.addWidget(self.main_window.rec_op_add_btn, 0)
        op_input_layout.addWidget(self.main_window.rec_op_remove_btn, 0)
        op_input_layout.addWidget(self.main_window.rec_op_up_btn, 0)
        op_input_layout.addWidget(self.main_window.rec_op_down_btn, 0)
        op_layout.addLayout(op_input_layout)

        item_group = QGroupBox("道具列表")
        set_group_box_icon(item_group, "item.png")
        item_layout = QVBoxLayout(item_group)
        item_layout.setContentsMargins(6, 6, 6, 6)
        self.main_window.rec_item_table = QTableWidget()
        self.main_window.rec_item_table.setColumnCount(2)
        self.main_window.rec_item_table.setHorizontalHeaderLabels(["道具名", "次数"])
        self.main_window.rec_item_table.setMinimumHeight(200)
        self.main_window.rec_item_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        item_layout.addWidget(self.main_window.rec_item_table)
        item_input_layout = QHBoxLayout()
        item_input_layout.setSpacing(4)
        self.main_window.rec_item_input = QLineEdit()
        self.main_window.rec_item_input.setPlaceholderText("道具名...")
        self.main_window.rec_item_input.setMinimumWidth(100)
        self.main_window.rec_item_charges_input = QSpinBox()
        self.main_window.rec_item_charges_input.setRange(1, 999)
        self.main_window.rec_item_charges_input.setValue(1)
        self.main_window.rec_item_charges_input.setFixedWidth(95)
        self.main_window.rec_item_add_btn = QPushButton("添加")
        self.main_window.rec_item_remove_btn = QPushButton("删除")
        self.main_window.rec_item_up_btn = QPushButton("上移")
        self.main_window.rec_item_down_btn = QPushButton("下移")
        for btn in (
            self.main_window.rec_item_add_btn,
            self.main_window.rec_item_remove_btn,
            self.main_window.rec_item_up_btn,
            self.main_window.rec_item_down_btn,
        ):
            btn.setFixedHeight(22)
            btn.setFixedWidth(45)
            btn.setStyleSheet("padding: 1px 4px; font-size: 11px;")
        item_input_layout.addWidget(self.main_window.rec_item_input, 1)
        item_input_layout.addWidget(self.main_window.rec_item_charges_input, 0)
        item_input_layout.addWidget(self.main_window.rec_item_add_btn, 0)
        item_input_layout.addWidget(self.main_window.rec_item_remove_btn, 0)
        item_input_layout.addWidget(self.main_window.rec_item_up_btn, 0)
        item_input_layout.addWidget(self.main_window.rec_item_down_btn, 0)
        item_layout.addLayout(item_input_layout)

        layout.addWidget(param_group)

        # 干员列表 + 道具列表并排
        list_layout = QHBoxLayout()
        list_layout.addWidget(op_group, 1)
        list_layout.addWidget(item_group, 1)
        layout.addLayout(list_layout)

        # 绑定录制器列表按钮
        self.main_window.rec_op_add_btn.clicked.connect(self._rec_add_operator)
        self.main_window.rec_op_remove_btn.clicked.connect(self._rec_remove_operator)
        self.main_window.rec_op_up_btn.clicked.connect(self._rec_move_op_up)
        self.main_window.rec_op_down_btn.clicked.connect(self._rec_move_op_down)
        self.main_window.rec_item_add_btn.clicked.connect(self._rec_add_item)
        self.main_window.rec_item_remove_btn.clicked.connect(self._rec_remove_item)
        self.main_window.rec_item_up_btn.clicked.connect(self._rec_move_item_up)
        self.main_window.rec_item_down_btn.clicked.connect(self._rec_move_item_down)

        # 干员/道具列表变化时自动保存配置
        self.main_window.rec_op_table.cellChanged.connect(self.main_window._save_config)

        # 录制参数变更自动保存配置
        self.main_window.rec_stage_code.textChanged.connect(self.main_window._save_config)
        self.main_window.rec_initial_operator_count.valueChanged.connect(self.main_window._save_config)
        self.main_window.rec_initial_item_count.valueChanged.connect(self.main_window._save_config)
        self.main_window.rec_initial_operator_count.valueChanged.connect(self._sync_recorder_counts)
        self.main_window.rec_initial_item_count.valueChanged.connect(self._sync_recorder_counts)
        self.main_window.rec_cost_tag.currentIndexChanged.connect(self.main_window._save_config)
        self.main_window.combo_rec_debug.item_changed.connect(self.main_window._save_config)
        self.main_window.rec_chk_support_op.stateChanged.connect(self.main_window._save_config)
        self.main_window.rec_chk_support_op.stateChanged.connect(self._sync_recorder_counts)
        self.main_window.rec_takeover_hotkey.textChanged.connect(self.main_window._save_config)
        self.main_window.rec_chk_probability_retry.stateChanged.connect(self.main_window._save_config)
        self.main_window.rec_loaded_script_path.textChanged.connect(self._update_loaded_script_status)

        # 录制执行
        exec_group = QGroupBox("录制执行")
        set_group_box_icon(exec_group, "battle.png")
        exec_layout = QVBoxLayout(exec_group)
        exec_layout.setContentsMargins(6, 6, 6, 6)
        exec_layout.setSpacing(6)

        btn_layout = QHBoxLayout()
        self.main_window.btn_rec_start = QPushButton("开始录制")
        self.main_window.btn_rec_start.setStyleSheet("background-color: #f44336; color: white;")
        self.main_window.btn_rec_start.clicked.connect(self._start_recording)
        btn_layout.addWidget(self.main_window.btn_rec_start)

        self.main_window.btn_rec_stop = QPushButton("停止录制")
        self.main_window.btn_rec_stop.setEnabled(False)
        self.main_window.btn_rec_stop.clicked.connect(self._on_rec_stop_clicked)
        btn_layout.addWidget(self.main_window.btn_rec_stop)

        self.main_window.btn_rec_save = QPushButton("导出脚本")
        self.main_window.btn_rec_save.setEnabled(False)
        self.main_window.btn_rec_save.clicked.connect(self._save_recording)
        btn_layout.addWidget(self.main_window.btn_rec_save)

        btn_layout.addStretch()
        exec_layout.addLayout(btn_layout)

        self.main_window.rec_status = QLabel("状态: 就绪")
        exec_layout.addWidget(self.main_window.rec_status)
        layout.addWidget(exec_group)

        layout.addStretch()

        # 与脚本执行 Tab 保持一致的勾选框样式：选中时用绿色填充，不显示默认对勾
        checkbox_green_fill_style = (
            "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #1a1a1a; border-radius: 3px; image: none; }"
            "QCheckBox::indicator:checked { background-color: #4CAF50; }"
        )
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(checkbox_green_fill_style)

    def _browse_loaded_script(self):
        path, _ = QFileDialog.getOpenFileName(
            self.main_window, "选择要装载的脚本", "", "JSON 文件 (*.json)"
        )
        if not path:
            return
        self.main_window.rec_loaded_script_path.setText(path)
        self._update_loaded_script_status()
        self.main_window._save_config()

    def _clear_loaded_script(self):
        self.main_window.rec_loaded_script_path.setText("")
        self._update_loaded_script_status()
        self.main_window._save_config()

    def _update_loaded_script_status(self):
        path = self.main_window.rec_loaded_script_path.text().strip()
        has_script = bool(path)
        if has_script:
            self.main_window.rec_loaded_script_status.setText(
                f"已装载 {Path(path).name}"
            )
        else:
            self.main_window.rec_loaded_script_status.setText("当前未装载脚本")
        # 没有装载脚本时禁用概率点配置
        self.main_window.rec_chk_probability_retry.setEnabled(has_script)
        if not has_script:
            self.main_window.rec_chk_probability_retry.setChecked(False)
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        if overlay is not None:
            overlay.set_script_status(
                f"已装载 {Path(path).name}" if path else "当前未装载脚本"
            )

    def _parse_recorder_operators(self) -> list:
        operators = []
        for i in range(self.main_window.rec_op_table.rowCount()):
            name_item = self.main_window.rec_op_table.item(i, 0)
            cost_item = self.main_window.rec_op_table.item(i, 1)
            if name_item is None:
                continue
            name = name_item.text().strip()
            if not name:
                continue
            cost = None
            if cost_item is not None:
                try:
                    cost = int(cost_item.text())
                except ValueError:
                    cost = None
            operators.append((name, cost))
        return operators

    def _parse_recorder_items(self) -> list:
        items = []
        for r in range(self.main_window.rec_item_table.rowCount()):
            name_item = self.main_window.rec_item_table.item(r, 0)
            charges_item = self.main_window.rec_item_table.item(r, 1)
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

    def _load_loaded_script(self, path: str):
        """加载要预执行的脚本，失败时弹出提示并返回 None。"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return ScriptModel.model_validate(data)
        except Exception as e:
            QMessageBox.critical(
                self.main_window,
                "装载脚本失败",
                f"无法读取脚本 {path}:\n{e}",
            )
            return None

    def _on_rec_takeover(self):
        """悬浮窗“手动接管”按钮回调。"""
        if self.main_window._recorder is not None:
            self.main_window._recorder.take_over()
            self.main_window.rec_status.setText("状态: 已请求手动接管")

    def _on_probability_retry_changed(self, state):
        """概率点配置勾选框状态变化：启用/禁用配置按钮。"""
        if isinstance(state, Qt.CheckState):
            checked = state == Qt.CheckState.Checked
        else:
            checked = state == Qt.CheckState.Checked.value
        self.main_window.btn_rec_probability_config.setEnabled(checked)

    def _open_probability_config(self):
        """打开概率点重试配置对话框。"""
        dialog = ProbabilityRetryConfigDialog(self.main_window, self._probability_retry_config)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._probability_retry_config = dialog.get_config()
            self.main_window._save_config()

    def _sync_recorder_counts(self):
        """录制参数中的初始干员/道具/助战变更时同步到运行中的录制器。"""
        recorder = getattr(self.main_window, "_recorder", None)
        if recorder is None:
            return
        try:
            recorder.set_initial_operator_count(
                self.main_window.rec_initial_operator_count.value()
            )
            recorder.set_initial_item_count(
                self.main_window.rec_initial_item_count.value()
            )
            recorder.set_support_count(
                1 if self.main_window.rec_chk_support_op.isChecked() else 0
            )
        except Exception as e:
            print(f"[recorder_tab] 同步录制器参数失败: {e}")

    def _start_recording(self):
        initial_operator_count = self.main_window.rec_initial_operator_count.value()
        if initial_operator_count <= 0:
            QMessageBox.warning(self.main_window, "参数错误", "请在“初始干员数量”中填写编队携带的干员数量（至少 1）。")
            return

        initial_item_count = self.main_window.rec_initial_item_count.value()

        stage_code = self.main_window.rec_stage_code.text().strip()
        if not stage_code:
            QMessageBox.warning(self.main_window, "参数错误", "请填写关卡代号（如 1-7）。")
            return

        if load_stage_dimensions(stage_code) is None:
            QMessageBox.warning(
                self.main_window,
                "关卡代号错误",
                f"未在 levels.json 中找到关卡代号 '{stage_code}'，请检查输入的关卡代号是否正确。",
            )
            return

        loaded_script_path = self.main_window.rec_loaded_script_path.text().strip()
        loaded_script = None
        if loaded_script_path:
            loaded_script = self._load_loaded_script(loaded_script_path)
            if loaded_script is None:
                return
            if loaded_script.stage_code != stage_code:
                QMessageBox.warning(
                    self.main_window,
                    "关卡不一致",
                    f"装载脚本的关卡为 {loaded_script.stage_code}，"
                    f"与录制参数中的 {stage_code} 不一致，请检查。",
                )
                return

            warning_parts = []
            script_operator_count = len(loaded_script.operators or [])
            script_item_count = len(loaded_script.items or [])
            if (
                script_operator_count != initial_operator_count
                or script_item_count != initial_item_count
            ):
                warning_parts.append(
                    f"装载脚本的初始数量与当前录制参数不一致，请检查。\n"
                    f"脚本：干员 {script_operator_count} 个，道具 {script_item_count} 个\n"
                    f"当前参数：干员 {initial_operator_count} 个，道具 {initial_item_count} 个\n"
                    f"若继续执行，部署栏点击位置可能与脚本预期不符。"
                )

            dir_issues = loaded_script.validate_deploy_directions()
            if dir_issues:
                dir_lines = []
                for time_ms, category, name in dir_issues:
                    s, f = self.main_window._ms_to_sf(time_ms)
                    dir_lines.append(f"{s}秒第{f}帧：{category} {name} 部署缺少方向")
                warning_parts.append(
                    "以下部署动作缺少方向参数（干员必须填写；道具/召唤物因半数以上部署有方向，视为需要方向）：\n"
                    + "\n".join(dir_lines)
                )

            if warning_parts:
                msg = QMessageBox(self.main_window)
                msg.setIcon(QMessageBox.Icon.Warning)
                msg.setWindowTitle("脚本校验警告")
                msg.setText("\n\n".join(warning_parts))
                execute_btn = msg.addButton("直接执行", QMessageBox.ButtonRole.AcceptRole)
                edit_btn = msg.addButton("返回编辑", QMessageBox.ButtonRole.RejectRole)
                msg.setDefaultButton(edit_btn)
                msg.exec()
                if msg.clickedButton() == edit_btn:
                    return

        # 如果已有悬浮窗/录制器，先清理掉，重新走完整流程
        self._cleanup_recorder_overlay()

        support_count = 1 if self.main_window.rec_chk_support_op.isChecked() else 0
        debug_keys = set(self.main_window.combo_rec_debug.checked_data())
        rec_debug = "recorder" in debug_keys
        rec_debug_cost_bar = "cost_bar" in debug_keys
        rec_debug_resolver = "resolver" in debug_keys
        rec_debug_screenshot = "screenshot" in debug_keys
        rec_debug_loaded_script = "loaded_script" in debug_keys
        rec_debug_skill_status = "skill_status" in debug_keys

        try:
            self.main_window._recorder_capture = WindowCapture(
                backend="mss",
                debug=rec_debug,
            )
        except Exception as e:
            QMessageBox.critical(self.main_window, "错误", f"窗口捕获初始化失败:\n{e}")
            return

        self._normal_overlay_started = False
        self._recorder_overlay_timer_mode = False
        self._recording_overlay_switch_at = 0.0

        overlay = InfoCollectionOverlay(debug=rec_debug)
        overlay.set_button_callbacks(
            start_callback=self._start_recording,
            stop_callback=self._stop_recording,
            takeover_callback=self._on_rec_takeover,
        )
        overlay.set_phase("正在初始化 OCR...")
        overlay.show()
        remove_dwm_glass_border(overlay)
        set_tool_window_style(overlay)
        set_window_topmost(overlay)
        QApplication.processEvents()

        ocr = None
        try:
            ocr = OCREngine(engine=None, debug=rec_debug, use_gpu=True)
        except Exception as e:
            overlay.close_overlay()
            self.main_window._recorder_overlay = None
            QMessageBox.critical(self.main_window, "错误", f"OCR 初始化失败:\n{e}")
            return

        self.main_window.showMinimized()
        if loaded_script is not None:
            overlay.set_phase("脚本操作中...")
        else:
            overlay.set_phase("请保持编队界面...")
        overlay.set_recording_state(True)
        QApplication.processEvents()

        def _resolver_log(line: str):
            self.main_window.log_text.append(line)
            QApplication.processEvents()

        probability_retry_enabled = self.main_window.rec_chk_probability_retry.isChecked()
        retry_cfg = self._probability_retry_config
        self.main_window._recorder = ActionRecorder(
            capture=self.main_window._recorder_capture,
            timer=self.main_window._region_timer,
            stage_code=stage_code,
            debug=rec_debug,
            debug_cost_bar=rec_debug_cost_bar,
            debug_resolver=rec_debug_resolver,
            debug_screenshot=rec_debug_screenshot,
            debug_loaded_script=rec_debug_loaded_script,
            debug_skill_status=rec_debug_skill_status,
            initial_operator_count=initial_operator_count,
            initial_item_count=initial_item_count,
            support_count=support_count,
            pause_key=action.pause_key(),
            takeover_hotkey=self.main_window.rec_takeover_hotkey.text().strip().upper() or "F9",
            matchstick_hotkeys=self.main_window.timer_tab._build_matchstick_hotkeys() or None,
            cost_bar_calibration_name=self.main_window.rec_cost_tag.currentData() or None,
            ocr=ocr,
            resolver_log_callback=_resolver_log if rec_debug_resolver else None,
            loaded_script=loaded_script,
            loaded_script_path=loaded_script_path,
            probability_retry_enabled=probability_retry_enabled,
            challenge_mode=retry_cfg.get("challenge_mode", False),
            sand_table=retry_cfg.get("sand_table", False),
            support_friend_index=retry_cfg.get("support_friend_index", 0),
            support_skill=retry_cfg.get("support_skill", 1),
            support_module=retry_cfg.get("support_module", 1),
        )
        def _on_takeover_mode_changed(show: bool):
            # 录制器回调可能来自非主线程（键盘监听/脚本执行线程），
            # 通过 QMetaObject.invokeMethod 保证 Qt UI 操作在主线程执行。
            try:
                QMetaObject.invokeMethod(
                    overlay,
                    "show_takeover_mode",
                    Qt.ConnectionType.QueuedConnection,
                    Q_ARG(bool, show),
                )
            except Exception as e:
                print(f"[recorder_tab] 切换接管模式失败: {e}")

        self.main_window._recorder.set_takeover_callback(_on_takeover_mode_changed)
        self.main_window._recorder.set_timer_adjusted_callback(
            self._on_executor_timer_adjusted
        )
        self.main_window._recorder_overlay = overlay
        self._update_loaded_script_status()
        self.main_window._recorder.start()

        self.main_window._recorder_poll_timer = QTimer(self.main_window)
        self.main_window._recorder_poll_timer.timeout.connect(self._poll_recorder_state)
        self.main_window._recorder_poll_timer.start(33)

        self.main_window.btn_rec_start.setEnabled(False)
        self.main_window.btn_rec_stop.setText("停止录制")
        self.main_window.btn_rec_stop.setEnabled(True)
        self.main_window.btn_rec_save.setEnabled(False)
        self.main_window.rec_status.setText("加载完毕，请进入作战...")

    def _cleanup_recorder_overlay(self):
        """清理现有录制器和悬浮窗（不保存），用于重新录制前重置。"""
        if hasattr(self.main_window, "_recorder_poll_timer") and self.main_window._recorder_poll_timer is not None:
            self.main_window._recorder_poll_timer.stop()
            self.main_window._recorder_poll_timer = None
        if self.main_window._recorder is not None:
            try:
                self.main_window._recorder.stop()
            except Exception:
                pass
            self.main_window._recorder = None
        self.main_window._recorder_capture = None
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        if overlay is not None:
            overlay.close_overlay()
            self.main_window._recorder_overlay = None

    def _poll_recorder_state(self):
        if self.main_window._recorder is None:
            return
        state = getattr(self.main_window._recorder, "_state", "")
        if self.main_window._recorder.is_stop_requested():
            self._stop_recording()
            return
        overlay = getattr(self.main_window, "_recorder_overlay", None)

        # 显示装载脚本执行异常（只显示一次）
        loaded_script_error = getattr(self.main_window._recorder, "_loaded_script_error", None)
        if loaded_script_error:
            error_summary = loaded_script_error.splitlines()[0]
            if len(error_summary) > 80:
                error_summary = error_summary[:77] + "..."
            if overlay is not None:
                overlay.set_phase(f"脚本执行出错: {error_summary}")
            self.main_window.rec_status.setText(f"状态: 脚本执行出错 - {error_summary}")
            self.main_window._recorder._loaded_script_error = None
            return

        debug_keys = set(self.main_window.combo_rec_debug.checked_data())
        debug = "recorder" in debug_keys or "loaded_script" in debug_keys
        if debug:
            print(f"[recorder_tab poll] state={state} started={getattr(self, '_normal_overlay_started', False)} timer_mode={getattr(self, '_recorder_overlay_timer_mode', False)} switch_at={getattr(self, '_recording_overlay_switch_at', 0):.3f} now={time.time():.3f}")
        if overlay is not None:
            if state == "WAITING_FOR_START":
                has_loaded_script = getattr(self.main_window._recorder, "loaded_script", None) is not None
                if has_loaded_script:
                    if debug:
                        print("[recorder_tab poll] set_phase 脚本操作中...")
                    overlay.set_phase("脚本操作中...")
                elif getattr(self.main_window._recorder, "is_squad_capture_done", lambda: False)():
                    if debug:
                        print("[recorder_tab poll] set_phase 识别完成，可以开始进入作战")
                    overlay.set_phase("识别完成，可以开始进入作战")
                else:
                    if debug:
                        print("[recorder_tab poll] set_phase 请保持编队界面...")
                    overlay.set_phase("请保持编队界面...")
            elif state == "IDLE":
                if not getattr(self, "_normal_overlay_started", False):
                    if debug:
                        print("[recorder_tab poll] set_phase 开始录制，F10停止录制")
                    overlay.set_phase("开始录制，F10停止录制")
                    self._normal_overlay_started = True
                    self._recording_overlay_switch_at = time.time() + 3
                elif time.time() >= self._recording_overlay_switch_at:
                    if debug:
                        print("[recorder_tab poll] switch to timer")
                    self._switch_recorder_overlay_to_timer()
            elif state == "EXECUTING_LOADED_SCRIPT":
                # 脚本执行阶段切换到计时显示，方便用户观察与调试
                self._recorder_overlay_timer_mode = True
                if debug:
                    print("[recorder_tab poll] switch to timer (loaded script)")
                self._update_recorder_overlay_time(overlay)
            elif state == "TRANSITIONING_TO_TAKEOVER":
                if debug:
                    print("[recorder_tab poll] set_phase 正在转为接管状态...")
                overlay.set_phase("正在转为接管状态...")
            # 计时显示一旦启用，就不应受状态机影响，保证拖拽/选方向时时间仍刷新
            if getattr(self, "_recorder_overlay_timer_mode", False):
                if debug:
                    print("[recorder_tab poll] update_recorder_overlay_time")
                self._update_recorder_overlay_time(overlay)
        if self.main_window._recorder.is_recording() and state == "IDLE" \
                and self.main_window.rec_status.text().startswith("状态: 录制器已加载"):
            self.main_window.rec_status.setText("状态: 录制中")

    def _switch_recorder_overlay_to_timer(self):
        """录制提示 5 秒后切换为计时显示。"""
        debug_keys = set(self.main_window.combo_rec_debug.checked_data())
        debug = "recorder" in debug_keys or "loaded_script" in debug_keys
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        if debug:
            print(f"[_switch_recorder_overlay_to_timer] overlay={overlay}")
        if overlay is None:
            return
        self._recorder_overlay_timer_mode = True
        if self.main_window._recorder is not None:
            elapsed = self.main_window._recorder.get_display_time_ms()
            timer = self.main_window._recorder.timer
        else:
            timer = self.main_window._region_timer
            elapsed = timer.get_elapsed_ms() if timer is not None and timer.is_started() else 0.0
        if debug:
            print(f"[_switch_recorder_overlay_to_timer] timer={timer} started={timer.is_started() if timer else None}")
        if timer is not None and timer.is_started():
            s, f = self.main_window._ms_to_sf_for_timer(elapsed)
            if debug:
                print(f"[_switch_recorder_overlay_to_timer] elapsed={elapsed:.1f} s={s} f={f}")
            overlay.set_time(
                s, f, elapsed,
                getattr(timer, "_rate", 1.0),
                timer.is_manual_paused() if hasattr(timer, "is_manual_paused") else False,
            )
        else:
            if debug:
                print("[_switch_recorder_overlay_to_timer] timer not started, show 0")
            overlay.set_time(0, 0)

    def _update_recorder_overlay_time(self, overlay):
        """刷新录制浮窗的计时显示。"""
        debug_keys = set(self.main_window.combo_rec_debug.checked_data())
        debug = "recorder" in debug_keys or "loaded_script" in debug_keys
        if self.main_window._recorder is not None:
            elapsed = self.main_window._recorder.get_display_time_ms()
            timer = self.main_window._recorder.timer
        else:
            timer = self.main_window._region_timer
            if timer is None:
                if debug:
                    print("[_update_recorder_overlay_time] timer is None")
                return
            elapsed = timer.get_elapsed_ms()
        s, f = self.main_window._ms_to_sf_for_timer(elapsed)
        if debug:
            print(f"[_update_recorder_overlay_time] elapsed={elapsed:.1f} s={s} f={f}")
        overlay.set_time(
            s, f, elapsed,
            getattr(timer, "_rate", 1.0),
            timer.is_manual_paused() if hasattr(timer, "is_manual_paused") else False,
        )

    def _on_executor_timer_adjusted(self):
        """executor 中计时器被主动调整后，立即同步刷新悬浮窗显示。

        使用 BlockingQueuedConnection 在主线程同步执行 set_time，避免跳帧
        中间帧被 33ms 轮询间隔跳过。
        """
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        if overlay is None:
            return
        if not getattr(self, "_recorder_overlay_timer_mode", False):
            return
        try:
            if self.main_window._recorder is not None:
                elapsed = self.main_window._recorder.get_display_time_ms()
                timer = self.main_window._recorder.timer
            else:
                timer = self.main_window._region_timer
                if timer is None:
                    return
                elapsed = timer.get_elapsed_ms()
            s, f = self.main_window._ms_to_sf_for_timer(elapsed)
            if QThread.currentThread() == self.main_window.thread():
                overlay.set_time(
                    s, f, elapsed,
                    getattr(timer, "_rate", 1.0),
                    timer.is_manual_paused() if hasattr(timer, "is_manual_paused") else False,
                )
            else:
                QMetaObject.invokeMethod(
                    overlay, "set_time",
                    Qt.ConnectionType.BlockingQueuedConnection,
                    Q_ARG(int, s),
                    Q_ARG(int, f),
                    Q_ARG(float, elapsed),
                    Q_ARG(float, getattr(timer, "_rate", 1.0)),
                    Q_ARG(bool, timer.is_manual_paused() if hasattr(timer, "is_manual_paused") else False),
                )
        except Exception as e:
            if "loaded_script" in set(self.main_window.combo_rec_debug.checked_data()):
                print(f"[_on_executor_timer_adjusted] 更新失败: {e}")

    def _on_rec_stop_clicked(self):
        """主窗口“停止录制/关闭悬浮窗”按钮的统一入口。"""
        if self.main_window._recorder is not None:
            self._stop_recording()
        elif getattr(self.main_window, "_recorder_overlay", None) is not None:
            self._close_recorder_overlay()

    def _close_recorder_overlay(self):
        """关闭悬浮窗并恢复主窗口。"""
        self._cleanup_recorder_overlay()
        self.main_window.showNormal()
        self.main_window.btn_rec_start.setEnabled(True)
        self.main_window.btn_rec_stop.setText("停止录制")
        self.main_window.btn_rec_stop.setEnabled(False)
        self.main_window.rec_status.setText("状态: 悬浮窗已关闭")

    def _auto_save_script(self, script, stage_code: str, loaded_script_path: str = "") -> Path:
        """自动保存脚本到 scripts/<stage_code>/ 目录，序号递增避免覆盖。

        当存在装载脚本时，保存为 <原脚本名>_new_001.json；
        否则按关卡代号保存为 <stage_code>_001.json。
        """
        scripts_dir = get_project_root() / "scripts" / stage_code
        scripts_dir.mkdir(parents=True, exist_ok=True)

        if loaded_script_path:
            base = Path(loaded_script_path).stem
            pattern = f"{base}_new_*.json"
        else:
            base = stage_code
            pattern = f"{base}_*.json"

        existing_nums = set()
        for p in scripts_dir.glob(pattern):
            try:
                suffix = p.stem.split("_")[-1]
                existing_nums.add(int(suffix))
            except ValueError:
                continue

        num = 1
        while num in existing_nums:
            num += 1

        if loaded_script_path:
            path = scripts_dir / f"{base}_new_{num:03d}.json"
        else:
            path = scripts_dir / f"{base}_{num:03d}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(script.model_dump(), f, ensure_ascii=False, indent=2)
        return path

    def _stop_recording(self):
        self._normal_overlay_started = False
        self._recorder_overlay_timer_mode = False
        self._recording_overlay_switch_at = 0.0
        if hasattr(self.main_window, "_recorder_poll_timer") and self.main_window._recorder_poll_timer is not None:
            self.main_window._recorder_poll_timer.stop()
            self.main_window._recorder_poll_timer = None

        overlay = getattr(self.main_window, "_recorder_overlay", None)
        stage_code = ""
        loaded_script_path = ""
        if self.main_window._recorder is not None:
            stage_code = getattr(self.main_window._recorder, "stage_code", "") or ""
            loaded_script_path = getattr(self.main_window._recorder, "loaded_script_path", "") or ""

        # 悬浮窗保持打开，显示生成中
        if overlay is not None:
            overlay.set_phase("脚本生成中...")
            overlay.set_recording_state(False)
            overlay.start_button.setEnabled(False)
            overlay.stop_button.setEnabled(False)
            QApplication.processEvents()

        script = None
        if self.main_window._recorder is not None:
            script = self.main_window._recorder.stop()
            self.main_window._last_recorded_script = script
            self.main_window._recorder = None
        self.main_window._recorder_capture = None

        saved_path = None
        if script is not None:
            try:
                saved_path = self._auto_save_script(script, stage_code, loaded_script_path)
            except Exception as e:
                if overlay is not None:
                    overlay.set_phase(f"保存失败: {e}")
                else:
                    QMessageBox.critical(self.main_window, "保存失败", str(e))

        # 更新主界面表格（后台）
        if script is not None:
            self._populate_recorder_tables(script)

        if overlay is not None:
            if saved_path is not None:
                overlay.set_save_path(saved_path)
            overlay.set_recording_state(False)
            overlay.start_button.setEnabled(True)
            overlay.stop_button.setEnabled(False)

        self.main_window.btn_rec_start.setEnabled(True)
        self.main_window.btn_rec_stop.setText("关闭悬浮窗")
        self.main_window.btn_rec_stop.setEnabled(True)
        self.main_window.btn_rec_save.setEnabled(True)

        action_count = len(script.actions) if script is not None else 0
        self.main_window.rec_status.setText(f"状态: 录制完成，共 {action_count} 个操作")

        if saved_path is not None:
            self.main_window.log_text.append(f"[录制] 自动保存脚本: {saved_path}")

    def _populate_recorder_tables(self, script):
        """用解析后的脚本回填录制界面的干员/道具列表，方便用户审阅与保存配置。"""
        self.main_window.rec_op_table.blockSignals(True)
        self.main_window.rec_op_table.setRowCount(0)
        for name in script.operators:
            row = self.main_window.rec_op_table.rowCount()
            self.main_window.rec_op_table.insertRow(row)
            self.main_window.rec_op_table.setItem(row, 0, QTableWidgetItem(name))
            self.main_window.rec_op_table.setItem(row, 1, QTableWidgetItem(""))
        self.main_window.rec_op_table.blockSignals(False)

        self.main_window.rec_item_table.blockSignals(True)
        self.main_window.rec_item_table.setRowCount(0)
        for item in script.items:
            row = self.main_window.rec_item_table.rowCount()
            self.main_window.rec_item_table.insertRow(row)
            self.main_window.rec_item_table.setItem(row, 0, QTableWidgetItem(item.name))
            self.main_window.rec_item_table.setItem(row, 1, QTableWidgetItem(str(item.charges)))
        self.main_window.rec_item_table.blockSignals(False)

        self.main_window._save_config()

    def _save_recording(self):
        if not hasattr(self.main_window, "_last_recorded_script") or self.main_window._last_recorded_script is None:
            return
        stage_code = getattr(self.main_window._last_recorded_script, "stage_code", "") or ""
        default_dir = get_project_root() / "scripts" / stage_code
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "保存录制脚本", str(default_dir), "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.main_window._last_recorded_script.model_dump(), f, ensure_ascii=False, indent=2)
            msg = f"已保存到:\n{path}"
            placeholders = [
                a.operator_name for a in self.main_window._last_recorded_script.actions
                if a.operator_name and a.operator_name.startswith("__")
            ]
            if placeholders:
                msg += "\n\n检测到以下占位名称，建议到编辑器中修正:\n" + ", ".join(sorted(set(placeholders)))
            QMessageBox.information(self.main_window, "保存成功", msg)
        except Exception as e:
            QMessageBox.critical(self.main_window, "保存失败", str(e))

    def _rec_add_operator(self):
        name = self.main_window.rec_op_input.text().strip()
        if name:
            cost = self.main_window.rec_op_cost_input.value()
            row = self.main_window.rec_op_table.rowCount()
            self.main_window.rec_op_table.insertRow(row)
            self.main_window.rec_op_table.setItem(row, 0, QTableWidgetItem(name))
            cost_text = str(cost) if cost > 0 else ""
            self.main_window.rec_op_table.setItem(row, 1, QTableWidgetItem(cost_text))
            self.main_window.rec_op_input.clear()
            self.main_window.rec_op_cost_input.setValue(0)
            self.main_window._save_config()

    def _rec_remove_operator(self):
        rows = sorted({idx.row() for idx in self.main_window.rec_op_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.main_window.rec_op_table.removeRow(r)
        self.main_window._save_config()

    def _rec_move_op_up(self):
        idx = self.main_window.rec_op_table.currentRow()
        if idx > 0:
            self._rec_swap_op_rows(idx, idx - 1)
            self.main_window.rec_op_table.setCurrentCell(idx - 1, 0)
            self.main_window._save_config()

    def _rec_move_op_down(self):
        idx = self.main_window.rec_op_table.currentRow()
        if 0 <= idx < self.main_window.rec_op_table.rowCount() - 1:
            self._rec_swap_op_rows(idx, idx + 1)
            self.main_window.rec_op_table.setCurrentCell(idx + 1, 0)
            self.main_window._save_config()

    def _rec_swap_op_rows(self, i: int, j: int):
        name_i = self.main_window.rec_op_table.item(i, 0).text()
        cost_i = self.main_window.rec_op_table.item(i, 1).text() if self.main_window.rec_op_table.item(i, 1) else ""
        name_j = self.main_window.rec_op_table.item(j, 0).text()
        cost_j = self.main_window.rec_op_table.item(j, 1).text() if self.main_window.rec_op_table.item(j, 1) else ""
        self.main_window.rec_op_table.setItem(i, 0, QTableWidgetItem(name_j))
        self.main_window.rec_op_table.setItem(i, 1, QTableWidgetItem(cost_j))
        self.main_window.rec_op_table.setItem(j, 0, QTableWidgetItem(name_i))
        self.main_window.rec_op_table.setItem(j, 1, QTableWidgetItem(cost_i))

    def _rec_add_item(self):
        name = self.main_window.rec_item_input.text().strip()
        if not name:
            return
        charges = self.main_window.rec_item_charges_input.value()
        row = self.main_window.rec_item_table.rowCount()
        self.main_window.rec_item_table.insertRow(row)
        self.main_window.rec_item_table.setItem(row, 0, QTableWidgetItem(name))
        self.main_window.rec_item_table.setItem(row, 1, QTableWidgetItem(str(charges)))
        self.main_window.rec_item_input.clear()
        self.main_window._save_config()

    def _rec_remove_item(self):
        rows = sorted({idx.row() for idx in self.main_window.rec_item_table.selectedIndexes()}, reverse=True)
        for r in rows:
            self.main_window.rec_item_table.removeRow(r)
        self.main_window._save_config()

    def _rec_move_item_up(self):
        idx = self.main_window.rec_item_table.currentRow()
        if idx > 0:
            self._rec_swap_item_rows(idx, idx - 1)
            self.main_window.rec_item_table.setCurrentCell(idx - 1, 0)
            self.main_window._save_config()

    def _rec_move_item_down(self):
        idx = self.main_window.rec_item_table.currentRow()
        if 0 <= idx < self.main_window.rec_item_table.rowCount() - 1:
            self._rec_swap_item_rows(idx, idx + 1)
            self.main_window.rec_item_table.setCurrentCell(idx + 1, 0)
            self.main_window._save_config()

    def _rec_swap_item_rows(self, i: int, j: int):
        name_i = self.main_window.rec_item_table.item(i, 0).text()
        charges_i = self.main_window.rec_item_table.item(i, 1).text()
        name_j = self.main_window.rec_item_table.item(j, 0).text()
        charges_j = self.main_window.rec_item_table.item(j, 1).text()
        self.main_window.rec_item_table.setItem(i, 0, QTableWidgetItem(name_j))
        self.main_window.rec_item_table.setItem(i, 1, QTableWidgetItem(charges_j))
        self.main_window.rec_item_table.setItem(j, 0, QTableWidgetItem(name_i))
        self.main_window.rec_item_table.setItem(j, 1, QTableWidgetItem(charges_i))

