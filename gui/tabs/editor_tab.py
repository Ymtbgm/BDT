import json
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QTableWidget,
    QTableWidgetItem, QAbstractItemView, QMessageBox,
    QFileDialog, QSpinBox, QComboBox, QListWidget,
    QSizePolicy, QGroupBox, QCheckBox,
)
from PyQt6.QtCore import Qt

from models.script_schema import ScriptModel, OperatorAction, ActionType, ItemInfo, SummonInfo


class EditorTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._dirty = False
        self._build_ui()

    def _mark_dirty(self):
        self._dirty = True

    def _clear_dirty(self):
        self._dirty = False

    def _build_ui(self):
        main_layout = QVBoxLayout(self)

        # 顶部横条：关卡信息 + 脚本管理
        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("关卡名"))
        self.main_window.stage_name_edit = QLineEdit()
        self.main_window.stage_name_edit.setMaximumWidth(120)
        top_bar.addWidget(self.main_window.stage_name_edit)

        top_bar.addWidget(QLabel("关卡代号"))
        self.main_window.stage_code_edit = QLineEdit()
        self.main_window.stage_code_edit.setMaximumWidth(80)
        top_bar.addWidget(self.main_window.stage_code_edit)

        top_bar.addWidget(QLabel("地图行数"))
        self.main_window.rows_spin = QSpinBox()
        self.main_window.rows_spin.setRange(1, 20)
        self.main_window.rows_spin.setValue(7)
        top_bar.addWidget(self.main_window.rows_spin)

        top_bar.addWidget(QLabel("地图列数"))
        self.main_window.cols_spin = QSpinBox()
        self.main_window.cols_spin.setRange(1, 20)
        self.main_window.cols_spin.setValue(9)
        top_bar.addWidget(self.main_window.cols_spin)

        top_bar.addStretch()

        main_layout.addLayout(top_bar)

        # 下方主区域
        content_layout = QHBoxLayout()

        # 左侧：干员 / 道具 / 召唤物
        lists_panel = QVBoxLayout()
        lists_panel.addWidget(QLabel("部署区干员初始列表"))
        self.main_window.operators_list = QListWidget()
        self.main_window.operators_list.setMinimumHeight(160)
        self.main_window.operators_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_window.operators_list.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.main_window.operators_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.main_window.operators_list.model().rowsMoved.connect(self._sync_operators_to_script)
        lists_panel.addWidget(self.main_window.operators_list)
        op_input_layout = QGridLayout()
        self.main_window.op_input = QLineEdit()
        self.main_window.op_input.setPlaceholderText("输入干员名...")
        self.main_window.btn_add_op = QPushButton("添加")
        self.main_window.btn_remove_op = QPushButton("删除")
        self.main_window.btn_up_op = QPushButton("上移")
        self.main_window.btn_down_op = QPushButton("下移")
        op_input_layout.addWidget(self.main_window.op_input, 0, 0, 1, 2)
        op_input_layout.addWidget(self.main_window.btn_add_op, 1, 0)
        op_input_layout.addWidget(self.main_window.btn_remove_op, 1, 1)
        op_input_layout.addWidget(self.main_window.btn_up_op, 2, 0)
        op_input_layout.addWidget(self.main_window.btn_down_op, 2, 1)
        lists_panel.addLayout(op_input_layout)

        lists_panel.addWidget(QLabel("部署区道具初始列表"))
        self.main_window.items_table = QTableWidget()
        self.main_window.items_table.setColumnCount(2)
        self.main_window.items_table.setHorizontalHeaderLabels(["道具名", "次数"])
        self.main_window.items_table.setMinimumHeight(100)
        self.main_window.items_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_window.items_table.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.main_window.items_table.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.main_window.items_table.setDragDropOverwriteMode(False)
        self.main_window.items_table.model().rowsMoved.connect(self._sync_items_to_script)
        self.main_window.items_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.main_window.items_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        lists_panel.addWidget(self.main_window.items_table)
        item_input_layout = QHBoxLayout()
        self.main_window.item_input = QLineEdit()
        self.main_window.item_input.setPlaceholderText("道具名...")
        self.main_window.item_charges_input = QSpinBox()
        self.main_window.item_charges_input.setRange(1, 999)
        self.main_window.item_charges_input.setValue(1)
        item_input_layout.addWidget(self.main_window.item_input)
        item_input_layout.addWidget(self.main_window.item_charges_input)
        lists_panel.addLayout(item_input_layout)

        item_btn_layout = QGridLayout()
        self.main_window.btn_add_item = QPushButton("添加")
        self.main_window.btn_remove_item = QPushButton("删除")
        self.main_window.btn_up_item = QPushButton("上移")
        self.main_window.btn_down_item = QPushButton("下移")
        item_btn_layout.addWidget(self.main_window.btn_add_item, 0, 0)
        item_btn_layout.addWidget(self.main_window.btn_remove_item, 0, 1)
        item_btn_layout.addWidget(self.main_window.btn_up_item, 1, 0)
        item_btn_layout.addWidget(self.main_window.btn_down_item, 1, 1)
        lists_panel.addLayout(item_btn_layout)

        lists_panel.addWidget(QLabel("特殊召唤物"))
        self.main_window.summons_table = QTableWidget()
        self.main_window.summons_table.setColumnCount(2)
        self.main_window.summons_table.setHorizontalHeaderLabels(["召唤物名", "费用"])
        self.main_window.summons_table.setMinimumHeight(90)
        self.main_window.summons_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.main_window.summons_table.setDragDropMode(QAbstractItemView.DragDropMode.NoDragDrop)
        self.main_window.summons_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.main_window.summons_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        lists_panel.addWidget(self.main_window.summons_table)
        summon_input_layout = QHBoxLayout()
        self.main_window.summon_input = QLineEdit()
        self.main_window.summon_input.setPlaceholderText("召唤物名...")
        self.main_window.summon_cost_input = QSpinBox()
        self.main_window.summon_cost_input.setRange(0, 999)
        self.main_window.summon_cost_input.setValue(5)
        summon_input_layout.addWidget(self.main_window.summon_input)
        summon_input_layout.addWidget(self.main_window.summon_cost_input)
        lists_panel.addLayout(summon_input_layout)

        summon_btn_layout = QGridLayout()
        self.main_window.btn_add_summon = QPushButton("添加")
        self.main_window.btn_remove_summon = QPushButton("删除")
        self.main_window.btn_up_summon = QPushButton("上移")
        self.main_window.btn_down_summon = QPushButton("下移")
        summon_btn_layout.addWidget(self.main_window.btn_add_summon, 0, 0)
        summon_btn_layout.addWidget(self.main_window.btn_remove_summon, 0, 1)
        summon_btn_layout.addWidget(self.main_window.btn_up_summon, 1, 0)
        summon_btn_layout.addWidget(self.main_window.btn_down_summon, 1, 1)
        lists_panel.addLayout(summon_btn_layout)

        lists_panel.addStretch()

        lists_widget = QWidget()
        lists_widget.setLayout(lists_panel)
        lists_widget.setMaximumWidth(220)
        content_layout.addWidget(lists_widget, 0)

        # 中间：时间轴列表
        mid_panel = QVBoxLayout()
        mid_panel.addWidget(QLabel("时间轴操作"))
        self.main_window.action_table = QTableWidget()
        self.main_window.action_table.setColumnCount(7)
        self.main_window.action_table.setHorizontalHeaderLabels(
            ["秒", "帧", "操作", "干员/道具", "格子", "方向", "装置"]
        )
        self.main_window.action_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.main_window.action_table.setEditTriggers(
            QAbstractItemView.EditTrigger.SelectedClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.DoubleClicked
        )
        mid_panel.addWidget(self.main_window.action_table)

        btn_layout = QHBoxLayout()
        self.main_window.btn_add = QPushButton("添加")
        self.main_window.btn_remove = QPushButton("删除")
        self.main_window.btn_up = QPushButton("上移")
        self.main_window.btn_down = QPushButton("下移")
        btn_layout.addWidget(self.main_window.btn_add)
        btn_layout.addWidget(self.main_window.btn_remove)
        btn_layout.addWidget(self.main_window.btn_up)
        btn_layout.addWidget(self.main_window.btn_down)
        mid_panel.addLayout(btn_layout)

        content_layout.addLayout(mid_panel, 3)

        # 右侧：操作详情
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("操作属性"))

        right_panel.addWidget(QLabel("时间"))
        time_layout = QHBoxLayout()
        self.main_window.edit_time_s = QSpinBox()
        self.main_window.edit_time_s.setRange(0, 999999)
        self.main_window.edit_time_s.setSuffix("秒")
        self.main_window.edit_time_s.setToolTip("秒")
        time_layout.addWidget(self.main_window.edit_time_s)
        self.main_window.edit_time_f = QSpinBox()
        self.main_window.edit_time_f.setRange(0, 29)
        self.main_window.edit_time_f.setSuffix("帧")
        self.main_window.edit_time_f.setToolTip("帧 (0-29)")
        time_layout.addWidget(self.main_window.edit_time_f)
        time_layout.addStretch()
        right_panel.addLayout(time_layout)

        self.main_window._action_labels = {
            ActionType.DEPLOY: "部署",
            ActionType.RETREAT: "撤退",
            ActionType.SKILL: "技能",
            ActionType.SPEED_UP: "加速",
            ActionType.SPEED_DOWN: "减速",
            ActionType.PAUSE: "暂停",
            ActionType.ADD_ITEM: "部署区新增道具",
            ActionType.ADD_SUMMON: "部署区新增召唤物",
        }
        self.main_window._action_labels_rev = {v: k for k, v in self.main_window._action_labels.items()}

        right_panel.addWidget(QLabel("操作类型"))
        self.main_window.combo_action = QComboBox()
        for act in ActionType:
            self.main_window.combo_action.addItem(self.main_window._action_labels.get(act, act.value), act)
        right_panel.addWidget(self.main_window.combo_action)

        right_panel.addWidget(QLabel("干员/道具名称"))
        self.main_window.combo_op = QComboBox()
        self.main_window.combo_op.setEditable(True)
        right_panel.addWidget(self.main_window.combo_op)

        self.main_window.grid_input_widget = QWidget()
        grid_layout = QVBoxLayout(self.main_window.grid_input_widget)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(QLabel("格子 (行,列)"))
        self.main_window.edit_grid = QLineEdit()
        grid_layout.addWidget(self.main_window.edit_grid)
        right_panel.addWidget(self.main_window.grid_input_widget)

        self.main_window.item_index_widget = QWidget()
        item_layout = QVBoxLayout(self.main_window.item_index_widget)
        item_layout.setContentsMargins(0, 0, 0, 0)
        item_layout.addWidget(QLabel("序号"))
        self.main_window.edit_item_index = QSpinBox()
        self.main_window.edit_item_index.setRange(0, 999)
        item_layout.addWidget(self.main_window.edit_item_index)
        item_layout.addWidget(QLabel("数量"))
        self.main_window.edit_item_charges = QSpinBox()
        self.main_window.edit_item_charges.setRange(1, 999)
        item_layout.addWidget(self.main_window.edit_item_charges)
        self.main_window.item_index_widget.hide()
        right_panel.addWidget(self.main_window.item_index_widget)

        self.main_window.summon_charges_widget = QWidget()
        summon_charges_layout = QVBoxLayout(self.main_window.summon_charges_widget)
        summon_charges_layout.setContentsMargins(0, 0, 0, 0)
        summon_charges_layout.addWidget(QLabel("数量"))
        self.main_window.edit_summon_charges = QSpinBox()
        self.main_window.edit_summon_charges.setRange(1, 999)
        self.main_window.edit_summon_charges.setValue(1)
        summon_charges_layout.addWidget(self.main_window.edit_summon_charges)
        self.main_window.summon_charges_widget.hide()
        right_panel.addWidget(self.main_window.summon_charges_widget)

        right_panel.addWidget(QLabel("方向"))
        self.main_window.edit_dir = QComboBox()
        self.main_window.edit_dir.addItems(["", "up", "down", "left", "right"])
        right_panel.addWidget(self.main_window.edit_dir)

        right_panel.addWidget(QLabel("是否为场上装置"))
        self.main_window.chk_is_object = QCheckBox("is_object")
        right_panel.addWidget(self.main_window.chk_is_object)

        right_panel.addStretch()

        right_top = QWidget()
        right_top.setLayout(right_panel)

        script_mgmt = QGroupBox("脚本管理")
        mgmt_layout = QVBoxLayout(script_mgmt)
        self.main_window.btn_new = QPushButton("新建")
        self.main_window.btn_open = QPushButton("打开")
        self.main_window.btn_save = QPushButton("保存")
        mgmt_layout.addWidget(self.main_window.btn_new)
        mgmt_layout.addWidget(self.main_window.btn_open)
        mgmt_layout.addWidget(self.main_window.btn_save)

        right_container = QVBoxLayout()
        right_container.addWidget(right_top, 1)
        right_container.addWidget(script_mgmt, 0)

        right_widget = QWidget()
        right_widget.setLayout(right_container)
        right_widget.setMaximumWidth(320)
        content_layout.addWidget(right_widget, 0)

        main_layout.addLayout(content_layout, 1)

        # 绑定事件
        self.main_window.btn_add.clicked.connect(self._add_action)
        self.main_window.btn_remove.clicked.connect(self._remove_action)
        self.main_window.btn_up.clicked.connect(self._move_up)
        self.main_window.btn_down.clicked.connect(self._move_down)
        self.main_window.btn_new.clicked.connect(self._new_script)
        self.main_window.btn_open.clicked.connect(self._open_script)
        self.main_window.btn_save.clicked.connect(self._save_script)
        self.main_window.action_table.itemSelectionChanged.connect(self._on_select)
        self.main_window.action_table.cellChanged.connect(self._on_cell_changed)
        self.main_window.combo_action.currentTextChanged.connect(self._on_action_type_changed)
        self.main_window.rows_spin.valueChanged.connect(self._update_script_meta)
        self.main_window.cols_spin.valueChanged.connect(self._update_script_meta)
        self.main_window.stage_name_edit.textChanged.connect(self._update_script_meta)
        self.main_window.stage_code_edit.textChanged.connect(self._update_script_meta)
        self.main_window.btn_add_op.clicked.connect(self._add_operator)
        self.main_window.btn_remove_op.clicked.connect(self._remove_operator)
        self.main_window.btn_up_op.clicked.connect(self._move_op_up)
        self.main_window.btn_down_op.clicked.connect(self._move_op_down)
        self.main_window.btn_add_item.clicked.connect(self._add_item)
        self.main_window.btn_remove_item.clicked.connect(self._remove_item)
        self.main_window.btn_up_item.clicked.connect(self._move_item_up)
        self.main_window.btn_down_item.clicked.connect(self._move_item_down)
        self.main_window.items_table.cellChanged.connect(self._on_item_cell_changed)

        self.main_window.btn_add_summon.clicked.connect(self._add_summon)
        self.main_window.btn_remove_summon.clicked.connect(self._remove_summon)
        self.main_window.btn_up_summon.clicked.connect(self._move_summon_up)
        self.main_window.btn_down_summon.clicked.connect(self._move_summon_down)
        self.main_window.summons_table.cellChanged.connect(self._on_summon_cell_changed)

        self.main_window.edit_time_s.valueChanged.connect(self._auto_apply_edit)
        self.main_window.edit_time_f.valueChanged.connect(self._auto_apply_edit)
        self.main_window.combo_action.currentIndexChanged.connect(self._auto_apply_edit)
        self.main_window.combo_op.currentTextChanged.connect(self._auto_apply_edit)
        self.main_window.edit_grid.textChanged.connect(self._auto_apply_edit)
        self.main_window.edit_item_index.valueChanged.connect(self._auto_apply_edit)
        self.main_window.edit_item_charges.valueChanged.connect(self._auto_apply_edit)
        self.main_window.edit_summon_charges.valueChanged.connect(self._auto_apply_edit)
        self.main_window.edit_dir.currentTextChanged.connect(self._auto_apply_edit)
        self.main_window.chk_is_object.stateChanged.connect(self._auto_apply_edit)

        self._refresh_table()

    def _update_script_meta(self):
        self.main_window.script.stage_name = self.main_window.stage_name_edit.text() or None
        self.main_window.script.stage_code = self.main_window.stage_code_edit.text() or None
        self.main_window.script.grid_rows = self.main_window.rows_spin.value()
        self.main_window.script.grid_cols = self.main_window.cols_spin.value()
        self._mark_dirty()

    def _sync_operators_to_script(self):
        self.main_window.script.operators = []
        for i in range(self.main_window.operators_list.count()):
            text = self.main_window.operators_list.item(i).text().strip()
            if text:
                self.main_window.script.operators.append(text)
        self._refresh_op_combo()
        self._mark_dirty()

    def _sync_items_to_script(self):
        self.main_window.script.items = []
        for i in range(self.main_window.items_table.rowCount()):
            name_item = self.main_window.items_table.item(i, 0)
            charges_item = self.main_window.items_table.item(i, 1)
            if name_item:
                name = name_item.text().strip()
                charges = 1
                if charges_item:
                    try:
                        charges = int(charges_item.text())
                    except ValueError:
                        charges = 1
                if name:
                    self.main_window.script.items.append(ItemInfo(name=name, charges=charges))
        self._refresh_op_combo()
        self._mark_dirty()

    def _add_operator(self):
        name = self.main_window.op_input.text().strip()
        if name:
            self.main_window.operators_list.addItem(name)
            self.main_window.op_input.clear()
            self._sync_operators_to_script()

    def _remove_operator(self):
        idx = self.main_window.operators_list.currentRow()
        if idx >= 0:
            self.main_window.operators_list.takeItem(idx)
            self._sync_operators_to_script()

    def _move_op_up(self):
        idx = self.main_window.operators_list.currentRow()
        if idx > 0:
            item = self.main_window.operators_list.takeItem(idx)
            self.main_window.operators_list.insertItem(idx - 1, item)
            self.main_window.operators_list.setCurrentRow(idx - 1)
            self._sync_operators_to_script()

    def _move_op_down(self):
        idx = self.main_window.operators_list.currentRow()
        if 0 <= idx < self.main_window.operators_list.count() - 1:
            item = self.main_window.operators_list.takeItem(idx)
            self.main_window.operators_list.insertItem(idx + 1, item)
            self.main_window.operators_list.setCurrentRow(idx + 1)
            self._sync_operators_to_script()

    def _add_item(self):
        name = self.main_window.item_input.text().strip()
        charges = self.main_window.item_charges_input.value()
        if name:
            row = self.main_window.items_table.rowCount()
            self.main_window.items_table.insertRow(row)
            self.main_window.items_table.setItem(row, 0, QTableWidgetItem(name))
            self.main_window.items_table.setItem(row, 1, QTableWidgetItem(str(charges)))
            self.main_window.item_input.clear()
            self.main_window.item_charges_input.setValue(1)
            self._sync_items_to_script()

    def _remove_item(self):
        idx = self.main_window.items_table.currentRow()
        if idx >= 0:
            self.main_window.items_table.removeRow(idx)
            self._sync_items_to_script()

    def _move_item_up(self):
        idx = self.main_window.items_table.currentRow()
        if idx > 0:
            self._swap_item_rows(idx, idx - 1)
            self.main_window.items_table.selectRow(idx - 1)
            self._sync_items_to_script()

    def _move_item_down(self):
        idx = self.main_window.items_table.currentRow()
        if 0 <= idx < self.main_window.items_table.rowCount() - 1:
            self._swap_item_rows(idx, idx + 1)
            self.main_window.items_table.selectRow(idx + 1)
            self._sync_items_to_script()

    def _swap_item_rows(self, i: int, j: int):
        self.main_window.items_table.blockSignals(True)
        for col in range(self.main_window.items_table.columnCount()):
            item_i = self.main_window.items_table.takeItem(i, col)
            item_j = self.main_window.items_table.takeItem(j, col)
            self.main_window.items_table.setItem(i, col, item_j)
            self.main_window.items_table.setItem(j, col, item_i)
        self.main_window.items_table.blockSignals(False)

    def _on_item_cell_changed(self, row, col):
        self._sync_items_to_script()

    def _refresh_summons_table(self):
        """按费用从低到高刷新特殊召唤物表格，同费用保持用户添加顺序。"""
        self.main_window.summons_table.blockSignals(True)
        self.main_window.summons_table.setRowCount(0)
        sorted_summons = sorted(
            enumerate(self.main_window.script.summons),
            key=lambda x: (x[1].cost, x[0]),
        )
        for display_row, (user_index, summon) in enumerate(sorted_summons):
            self.main_window.summons_table.insertRow(display_row)
            name_item = QTableWidgetItem(summon.name)
            name_item.setData(Qt.ItemDataRole.UserRole, user_index)
            cost_item = QTableWidgetItem(str(summon.cost))
            cost_item.setData(Qt.ItemDataRole.UserRole, user_index)
            self.main_window.summons_table.setItem(display_row, 0, name_item)
            self.main_window.summons_table.setItem(display_row, 1, cost_item)
        self.main_window.summons_table.blockSignals(False)
        self._refresh_op_combo()

    def _add_summon(self):
        name = self.main_window.summon_input.text().strip()
        cost = self.main_window.summon_cost_input.value()
        if name:
            self.main_window.script.summons.append(SummonInfo(name=name, cost=cost))
            self.main_window.summon_input.clear()
            self.main_window.summon_cost_input.setValue(5)
            self._mark_dirty()
            self._refresh_summons_table()

    def _remove_summon(self):
        idx = self.main_window.summons_table.currentRow()
        if idx >= 0:
            user_index = self.main_window.summons_table.item(idx, 0).data(Qt.ItemDataRole.UserRole)
            if 0 <= user_index < len(self.main_window.script.summons):
                del self.main_window.script.summons[user_index]
            self._mark_dirty()
            self._refresh_summons_table()

    def _move_summon_up(self):
        idx = self.main_window.summons_table.currentRow()
        if idx >= 0:
            user_index = self.main_window.summons_table.item(idx, 0).data(Qt.ItemDataRole.UserRole)
            if user_index > 0:
                summons = self.main_window.script.summons
                summons[user_index], summons[user_index - 1] = summons[user_index - 1], summons[user_index]
            self._mark_dirty()
            self._refresh_summons_table()

    def _move_summon_down(self):
        idx = self.main_window.summons_table.currentRow()
        if idx >= 0:
            user_index = self.main_window.summons_table.item(idx, 0).data(Qt.ItemDataRole.UserRole)
            if user_index < len(self.main_window.script.summons) - 1:
                summons = self.main_window.script.summons
                summons[user_index], summons[user_index + 1] = summons[user_index + 1], summons[user_index]
            self._mark_dirty()
            self._refresh_summons_table()

    def _swap_summon_rows(self, i: int, j: int):
        """保留兼容旧代码的交换行辅助方法，当前已不再使用。"""
        self.main_window.summons_table.blockSignals(True)
        for col in range(self.main_window.summons_table.columnCount()):
            item_i = self.main_window.summons_table.takeItem(i, col)
            item_j = self.main_window.summons_table.takeItem(j, col)
            self.main_window.summons_table.setItem(i, col, item_j)
            self.main_window.summons_table.setItem(j, col, item_i)
        self.main_window.summons_table.blockSignals(False)

    def _on_summon_cell_changed(self, row, col):
        item = self.main_window.summons_table.item(row, col)
        if item is None:
            return
        user_index = item.data(Qt.ItemDataRole.UserRole)
        if not (0 <= user_index < len(self.main_window.script.summons)):
            return
        name_item = self.main_window.summons_table.item(row, 0)
        cost_item = self.main_window.summons_table.item(row, 1)
        name = name_item.text().strip() if name_item else ""
        cost = 0
        if cost_item:
            try:
                cost = int(cost_item.text())
            except ValueError:
                cost = 0
        self.main_window.script.summons[user_index] = SummonInfo(name=name, cost=cost)
        self._mark_dirty()
        self._refresh_summons_table()

    def _refresh_op_combo(self):
        current = self.main_window.combo_op.currentText()
        self.main_window.combo_op.blockSignals(True)
        self.main_window.combo_op.clear()
        for op in self.main_window.script.operators:
            self.main_window.combo_op.addItem(op)
        if self.main_window.script.items:
            for item in self.main_window.script.items:
                self.main_window.combo_op.addItem(item.name)
        if self.main_window.script.summons:
            for summon in self.main_window.script.summons:
                self.main_window.combo_op.addItem(summon.name)
        idx = self.main_window.combo_op.findText(current)
        if idx >= 0:
            self.main_window.combo_op.setCurrentIndex(idx)
        else:
            self.main_window.combo_op.setEditText(current)
        self.main_window.combo_op.blockSignals(False)

    def _refresh_table(self):
        self.main_window.action_table.blockSignals(True)
        selected_time = None
        selected_action = None
        selected_op = None
        idx = self.main_window.action_table.currentRow()
        if 0 <= idx < len(self.main_window.script.actions):
            selected_time = self.main_window.script.actions[idx].time_ms
            selected_action = self.main_window.script.actions[idx].action
            selected_op = self.main_window.script.actions[idx].operator_name

        self.main_window.script.sort_actions()
        self.main_window.action_table.setRowCount(len(self.main_window.script.actions))
        new_idx = -1
        for i, act in enumerate(self.main_window.script.actions):
            s, f = self.main_window._ms_to_sf(act.time_ms)
            self.main_window.action_table.setItem(i, 0, QTableWidgetItem(str(s)))
            self.main_window.action_table.setItem(i, 1, QTableWidgetItem(str(f)))
            action_text = self.main_window._action_labels.get(act.action, act.action.value if act.action else "")
            self.main_window.action_table.setItem(i, 2, QTableWidgetItem(action_text))
            self.main_window.action_table.setItem(i, 3, QTableWidgetItem(act.operator_name or ""))
            grid = act.grid
            if act.action == ActionType.ADD_SUMMON:
                grid_str = ""
            elif (
                grid is None
                and act.action in (ActionType.RETREAT, ActionType.SKILL)
                and act.operator_name
            ):
                grid = self._get_default_grid_for_action(i, act.operator_name)
                try:
                    grid_str = f"{grid[0]},{grid[1]}" if grid else ""
                except Exception:
                    grid_str = ""
            else:
                try:
                    grid_str = f"{grid[0]},{grid[1]}" if grid else ""
                except Exception:
                    grid_str = ""
            self.main_window.action_table.setItem(i, 4, QTableWidgetItem(grid_str))
            self.main_window.action_table.setItem(i, 5, QTableWidgetItem(act.direction or ""))
            self.main_window.action_table.setItem(i, 6, QTableWidgetItem("是" if act.is_object else ""))
            if (
                act.time_ms == selected_time
                and act.action == selected_action
                and act.operator_name == selected_op
            ):
                new_idx = i

        self.main_window.action_table.blockSignals(False)
        if new_idx >= 0:
            self.main_window.action_table.selectRow(new_idx)

    def _get_default_grid_for_action(self, action_idx: int, operator_name: str) -> Optional[Tuple[int, int]]:
        for prev_act in reversed(self.main_window.script.actions[:action_idx]):
            if (
                prev_act.action == ActionType.DEPLOY
                and prev_act.operator_name == operator_name
            ):
                return prev_act.grid
        return None

    def _on_select(self):
        if self.main_window._applying_edit:
            return
        self.main_window._selecting = True
        try:
            idx = self.main_window.action_table.currentRow()
            if 0 <= idx < len(self.main_window.script.actions):
                act = self.main_window.script.actions[idx]
                s, f = self.main_window._ms_to_sf(act.time_ms)
                self.main_window.edit_time_s.setValue(s)
                self.main_window.edit_time_f.setValue(f)
                self.main_window.combo_action.setCurrentText(self.main_window._action_labels.get(act.action, act.action.value if act.action else ""))
                self.main_window.combo_op.setEditText(act.operator_name or "")
                if act.action == ActionType.ADD_ITEM:
                    if act.grid:
                        self.main_window.edit_item_index.setValue(act.grid[0])
                        self.main_window.edit_item_charges.setValue(act.grid[1])
                    else:
                        self.main_window.edit_item_index.setValue(0)
                        self.main_window.edit_item_charges.setValue(1)
                elif act.action == ActionType.ADD_SUMMON:
                    if act.grid:
                        self.main_window.edit_summon_charges.setValue(max(1, act.grid[0]))
                    else:
                        self.main_window.edit_summon_charges.setValue(1)
                else:
                    grid = act.grid
                    if (
                        grid is None
                        and act.action in (ActionType.RETREAT, ActionType.SKILL)
                        and act.operator_name
                    ):
                        grid = self._get_default_grid_for_action(idx, act.operator_name)
                    self.main_window.edit_grid.setText(f"{grid[0]},{grid[1]}" if grid else "")
                self.main_window.edit_dir.setCurrentText(act.direction or "")
                self.main_window.chk_is_object.setChecked(act.is_object)
                self._on_action_type_changed()
        finally:
            self.main_window._selecting = False

    def _on_cell_changed(self, row, col):
        if not (0 <= row < len(self.main_window.script.actions)):
            return
        act = self.main_window.script.actions[row]
        val = self.main_window.action_table.item(row, col).text().strip()

        if col in (0, 1):
            try:
                s = int(self.main_window.action_table.item(row, 0).text().strip())
                f = int(self.main_window.action_table.item(row, 1).text().strip())
                act.time_ms = self.main_window._sf_to_ms(s, f)
            except ValueError:
                pass
        elif col == 2:
            mapped = self.main_window._action_labels_rev.get(val, val)
            try:
                act.action = ActionType(mapped)
            except ValueError:
                pass
        elif col == 3:
            act.operator_name = val or None
        elif col == 4:
            if act.action == ActionType.ADD_SUMMON:
                return
            val = self.main_window._normalize_grid_text(val)
            if val:
                try:
                    r, c = map(int, val.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
        elif col == 5:
            act.direction = val or None
        elif col == 6:
            act.is_object = val == "是"

        self._mark_dirty()
        if col in (0, 1):
            self._refresh_table()
        elif self.main_window.action_table.currentRow() == row:
            self._on_select()

    def _on_action_type_changed(self):
        act = self.main_window.combo_action.currentData()
        if act == ActionType.DEPLOY:
            self.main_window.grid_input_widget.show()
            self.main_window.item_index_widget.hide()
            self.main_window.summon_charges_widget.hide()
            self.main_window.edit_dir.setEnabled(True)
            self.main_window.combo_op.setEnabled(True)
            self.main_window.chk_is_object.setEnabled(True)
        elif act in (ActionType.RETREAT, ActionType.SKILL):
            self.main_window.grid_input_widget.show()
            self.main_window.item_index_widget.hide()
            self.main_window.summon_charges_widget.hide()
            self.main_window.edit_dir.setEnabled(False)
            self.main_window.combo_op.setEnabled(True)
            self.main_window.chk_is_object.setEnabled(True)
        elif act == ActionType.ADD_ITEM:
            self.main_window.grid_input_widget.hide()
            self.main_window.item_index_widget.show()
            self.main_window.summon_charges_widget.hide()
            self.main_window.edit_dir.setEnabled(False)
            self.main_window.combo_op.setEnabled(True)
            self.main_window.chk_is_object.setEnabled(False)
        elif act == ActionType.ADD_SUMMON:
            self.main_window.grid_input_widget.hide()
            self.main_window.item_index_widget.hide()
            self.main_window.summon_charges_widget.show()
            self.main_window.edit_dir.setEnabled(False)
            self.main_window.combo_op.setEnabled(True)
            self.main_window.chk_is_object.setEnabled(False)
        else:
            self.main_window.grid_input_widget.show()
            self.main_window.item_index_widget.hide()
            self.main_window.summon_charges_widget.hide()
            self.main_window.edit_dir.setEnabled(False)
            self.main_window.combo_op.setEnabled(False)
            self.main_window.chk_is_object.setEnabled(False)

    def _apply_edit(self):
        idx = self.main_window.action_table.currentRow()
        if not (0 <= idx < len(self.main_window.script.actions)):
            return
        act = self.main_window.script.actions[idx]
        act.time_ms = self.main_window._sf_to_ms(self.main_window.edit_time_s.value(), self.main_window.edit_time_f.value())
        act.action = self.main_window.combo_action.currentData()

        if act.action == ActionType.DEPLOY:
            act.operator_name = self.main_window.combo_op.currentText() or None
            grid_text = self.main_window._normalize_grid_text(self.main_window.edit_grid.text())
            if grid_text:
                try:
                    r, c = map(int, grid_text.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
            act.direction = self.main_window.edit_dir.currentText() or None
            act.is_object = self.main_window.chk_is_object.isChecked()
        elif act.action in (ActionType.RETREAT, ActionType.SKILL):
            act.operator_name = self.main_window.combo_op.currentText() or None
            grid_text = self.main_window._normalize_grid_text(self.main_window.edit_grid.text())
            if grid_text:
                try:
                    r, c = map(int, grid_text.split(","))
                    act.grid = (r, c)
                except ValueError:
                    act.grid = None
            else:
                act.grid = None
            act.direction = None
            act.is_object = self.main_window.chk_is_object.isChecked()
        elif act.action == ActionType.ADD_ITEM:
            act.operator_name = self.main_window.combo_op.currentText() or None
            act.grid = (self.main_window.edit_item_index.value(), self.main_window.edit_item_charges.value())
            act.direction = None
            act.is_object = False
        elif act.action == ActionType.ADD_SUMMON:
            act.operator_name = self.main_window.combo_op.currentText() or None
            act.grid = (self.main_window.edit_summon_charges.value(), 0)
            act.direction = None
            act.is_object = False
        else:
            act.operator_name = None
            act.grid = None
            act.direction = None
            act.is_object = False

        self._refresh_table()

    def _auto_apply_edit(self):
        if self.main_window._applying_edit or self.main_window._selecting:
            return
        self.main_window._applying_edit = True
        try:
            self._apply_edit()
            self._mark_dirty()
        finally:
            self.main_window._applying_edit = False

    def _new_script(self):
        if self._dirty:
            reply = QMessageBox.question(
                self.main_window,
                "确认新建",
                "结果尚未保存，是否要新建？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        self.main_window.script = ScriptModel(grid_rows=7, grid_cols=9)
        for w in (self.main_window.stage_name_edit, self.main_window.stage_code_edit, self.main_window.rows_spin, self.main_window.cols_spin):
            w.blockSignals(True)
        self.main_window.stage_name_edit.clear()
        self.main_window.stage_code_edit.clear()
        self.main_window.rows_spin.setValue(7)
        self.main_window.cols_spin.setValue(9)
        for w in (self.main_window.stage_name_edit, self.main_window.stage_code_edit, self.main_window.rows_spin, self.main_window.cols_spin):
            w.blockSignals(False)
        self.main_window.operators_list.clear()
        self.main_window.items_table.setRowCount(0)
        self.main_window.summons_table.setRowCount(0)
        self._clear_dirty()
        self._refresh_op_combo()
        self._refresh_table()

    def _open_script(self):
        if self._dirty:
            reply = QMessageBox.question(
                self.main_window,
                "确认打开",
                "结果尚未保存，是否要打开其他脚本？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        path, _ = QFileDialog.getOpenFileName(self.main_window, "打开脚本", "", "JSON (*.json)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.main_window.script = ScriptModel(**data)
        except Exception as e:
            QMessageBox.critical(self.main_window, "打开失败", f"脚本格式错误或解析失败:\n{e}")
            return

        for w in (self.main_window.stage_name_edit, self.main_window.stage_code_edit, self.main_window.rows_spin, self.main_window.cols_spin):
            w.blockSignals(True)
        self.main_window.stage_name_edit.setText(self.main_window.script.stage_name or "")
        self.main_window.stage_code_edit.setText(self.main_window.script.stage_code or "")
        self.main_window.rows_spin.setValue(self.main_window.script.grid_rows)
        self.main_window.cols_spin.setValue(self.main_window.script.grid_cols)
        for w in (self.main_window.stage_name_edit, self.main_window.stage_code_edit, self.main_window.rows_spin, self.main_window.cols_spin):
            w.blockSignals(False)

        self.main_window.operators_list.clear()
        for op in self.main_window.script.operators:
            self.main_window.operators_list.addItem(op)

        self.main_window.items_table.blockSignals(True)
        self.main_window.items_table.setRowCount(0)
        for item in self.main_window.script.items:
            row = self.main_window.items_table.rowCount()
            self.main_window.items_table.insertRow(row)
            self.main_window.items_table.setItem(row, 0, QTableWidgetItem(item.name))
            self.main_window.items_table.setItem(row, 1, QTableWidgetItem(str(item.charges)))
        self.main_window.items_table.blockSignals(False)

        self._refresh_summons_table()
        self._clear_dirty()
        self._refresh_op_combo()
        self._refresh_table()

    def _save_script(self):
        path, _ = QFileDialog.getSaveFileName(self.main_window, "保存脚本", "", "JSON (*.json)")
        if path:
            if not path.endswith(".json"):
                path += ".json"
            self.main_window.script.sort_actions()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.main_window.script.model_dump(), f, ensure_ascii=False, indent=2)
            self._clear_dirty()
            QMessageBox.information(self.main_window, "保存成功", f"已保存到:\n{path}")

    def _add_action(self):
        self._apply_edit()
        max_time_ms = max((a.time_ms for a in self.main_window.script.actions), default=0)
        act = OperatorAction(time_ms=max_time_ms, action=ActionType.DEPLOY)
        self.main_window.script.actions.append(act)
        self._mark_dirty()
        self._refresh_table()
        new_idx = len(self.main_window.script.actions) - 1
        self.main_window.action_table.selectRow(new_idx)
        self.main_window.action_table.scrollToItem(
            self.main_window.action_table.item(new_idx, 0),
            QAbstractItemView.ScrollHint.EnsureVisible,
        )

    def _remove_action(self):
        self._apply_edit()
        idx = self.main_window.action_table.currentRow()
        if 0 <= idx < len(self.main_window.script.actions):
            del self.main_window.script.actions[idx]
            self._mark_dirty()
            self._refresh_table()
            if idx < len(self.main_window.script.actions):
                self.main_window.action_table.selectRow(idx)
            elif self.main_window.script.actions:
                self.main_window.action_table.selectRow(len(self.main_window.script.actions) - 1)

    def _move_up(self):
        self._apply_edit()
        idx = self.main_window.action_table.currentRow()
        if idx > 0:
            self.main_window.script.actions[idx], self.main_window.script.actions[idx - 1] = (
                self.main_window.script.actions[idx - 1],
                self.main_window.script.actions[idx],
            )
            self._mark_dirty()
            self._refresh_table()
            self.main_window.action_table.selectRow(idx - 1)

    def _move_down(self):
        self._apply_edit()
        idx = self.main_window.action_table.currentRow()
        if 0 <= idx < len(self.main_window.script.actions) - 1:
            self.main_window.script.actions[idx], self.main_window.script.actions[idx + 1] = (
                self.main_window.script.actions[idx + 1],
                self.main_window.script.actions[idx],
            )
            self._mark_dirty()
            self._refresh_table()
            self.main_window.action_table.selectRow(idx + 1)
