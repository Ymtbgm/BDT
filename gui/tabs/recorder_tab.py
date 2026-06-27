import json

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox,
    QGroupBox, QListWidget, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QFileDialog,
)
from PyQt6.QtCore import Qt, QTimer

from core.capture import WindowCapture
from core.recorder import ActionRecorder
from models.script_schema import ItemInfo


class RecorderTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 参数区域
        param_group = QGroupBox("录制参数")
        param_layout = QGridLayout()

        param_layout.addWidget(QLabel("关卡名称:"), 0, 0)
        self.main_window.rec_stage_name = QLineEdit()
        self.main_window.rec_stage_name.setPlaceholderText("如 1-7")
        param_layout.addWidget(self.main_window.rec_stage_name, 0, 1)

        param_layout.addWidget(QLabel("关卡代号:"), 0, 2)
        self.main_window.rec_stage_code = QLineEdit()
        param_layout.addWidget(self.main_window.rec_stage_code, 0, 3)

        param_layout.addWidget(QLabel("地图行数:"), 1, 0)
        self.main_window.rec_grid_rows = QSpinBox()
        self.main_window.rec_grid_rows.setRange(1, 50)
        self.main_window.rec_grid_rows.setValue(7)
        param_layout.addWidget(self.main_window.rec_grid_rows, 1, 1)

        param_layout.addWidget(QLabel("地图列数:"), 1, 2)
        self.main_window.rec_grid_cols = QSpinBox()
        self.main_window.rec_grid_cols.setRange(1, 50)
        self.main_window.rec_grid_cols.setValue(9)
        param_layout.addWidget(self.main_window.rec_grid_cols, 1, 3)

        self.main_window.rec_chk_debug = QCheckBox("Debug 模式")
        param_layout.addWidget(self.main_window.rec_chk_debug, 2, 0, 1, 4)

        # 干员列表 + 道具列表并排
        op_group = QGroupBox("干员列表")
        op_layout = QVBoxLayout(op_group)
        self.main_window.rec_op_list = QListWidget()
        self.main_window.rec_op_list.setMinimumHeight(100)
        op_layout.addWidget(self.main_window.rec_op_list)
        op_input_layout = QHBoxLayout()
        self.main_window.rec_op_input = QLineEdit()
        self.main_window.rec_op_input.setPlaceholderText("输入干员名...")
        self.main_window.rec_op_add_btn = QPushButton("添加")
        self.main_window.rec_op_remove_btn = QPushButton("删除")
        op_input_layout.addWidget(self.main_window.rec_op_input)
        op_input_layout.addWidget(self.main_window.rec_op_add_btn)
        op_input_layout.addWidget(self.main_window.rec_op_remove_btn)
        op_layout.addLayout(op_input_layout)

        item_group = QGroupBox("道具列表")
        item_layout = QVBoxLayout(item_group)
        self.main_window.rec_item_table = QTableWidget()
        self.main_window.rec_item_table.setColumnCount(2)
        self.main_window.rec_item_table.setHorizontalHeaderLabels(["道具名", "次数"])
        self.main_window.rec_item_table.setMinimumHeight(100)
        self.main_window.rec_item_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        item_layout.addWidget(self.main_window.rec_item_table)
        item_input_layout = QHBoxLayout()
        self.main_window.rec_item_input = QLineEdit()
        self.main_window.rec_item_input.setPlaceholderText("道具名...")
        self.main_window.rec_item_charges_input = QSpinBox()
        self.main_window.rec_item_charges_input.setRange(1, 999)
        self.main_window.rec_item_charges_input.setValue(1)
        self.main_window.rec_item_add_btn = QPushButton("添加")
        self.main_window.rec_item_remove_btn = QPushButton("删除")
        item_input_layout.addWidget(self.main_window.rec_item_input)
        item_input_layout.addWidget(self.main_window.rec_item_charges_input)
        item_input_layout.addWidget(self.main_window.rec_item_add_btn)
        item_input_layout.addWidget(self.main_window.rec_item_remove_btn)
        item_layout.addLayout(item_input_layout)

        list_layout = QHBoxLayout()
        list_layout.addWidget(op_group, 1)
        list_layout.addWidget(item_group, 1)
        param_layout.addLayout(list_layout, 3, 0, 1, 4)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # 绑定录制器列表按钮
        self.main_window.rec_op_add_btn.clicked.connect(self._rec_add_operator)
        self.main_window.rec_op_remove_btn.clicked.connect(self._rec_remove_operator)
        self.main_window.rec_item_add_btn.clicked.connect(self._rec_add_item)
        self.main_window.rec_item_remove_btn.clicked.connect(self._rec_remove_item)

        # 录制参数变更自动保存配置
        self.main_window.rec_stage_name.textChanged.connect(self.main_window._save_config)
        self.main_window.rec_stage_code.textChanged.connect(self.main_window._save_config)
        self.main_window.rec_grid_rows.valueChanged.connect(self.main_window._save_config)
        self.main_window.rec_grid_cols.valueChanged.connect(self.main_window._save_config)
        self.main_window.rec_chk_debug.stateChanged.connect(self.main_window._save_config)

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
        self.main_window.btn_rec_start = QPushButton("开始录制")
        self.main_window.btn_rec_start.setStyleSheet("background-color: #f44336; color: white;")
        self.main_window.btn_rec_start.clicked.connect(self._start_recording)
        btn_layout.addWidget(self.main_window.btn_rec_start)

        self.main_window.btn_rec_stop = QPushButton("停止录制")
        self.main_window.btn_rec_stop.setEnabled(False)
        self.main_window.btn_rec_stop.clicked.connect(self._stop_recording)
        btn_layout.addWidget(self.main_window.btn_rec_stop)

        self.main_window.btn_rec_save = QPushButton("导出脚本")
        self.main_window.btn_rec_save.setEnabled(False)
        self.main_window.btn_rec_save.clicked.connect(self._save_recording)
        btn_layout.addWidget(self.main_window.btn_rec_save)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.main_window.rec_status = QLabel("状态: 就绪")
        layout.addWidget(self.main_window.rec_status)

        layout.addStretch()

    def _parse_recorder_operators(self) -> list:
        return [
            self.main_window.rec_op_list.item(i).text()
            for i in range(self.main_window.rec_op_list.count())
            if self.main_window.rec_op_list.item(i).text().strip()
        ]

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

    def _start_recording(self):
        try:
            self.main_window._recorder_capture = WindowCapture(backend="mss")
        except Exception as e:
            QMessageBox.critical(self.main_window, "错误", f"窗口捕获初始化失败:\n{e}")
            return

        operators = self._parse_recorder_operators()
        items = self._parse_recorder_items()
        grid_rows = self.main_window.rec_grid_rows.value()
        grid_cols = self.main_window.rec_grid_cols.value()
        stage_name = self.main_window.rec_stage_name.text().strip() or None
        stage_code = self.main_window.rec_stage_code.text().strip() or None

        self.main_window._recorder = ActionRecorder(
            capture=self.main_window._recorder_capture,
            timer=self.main_window._region_timer,
            operators=operators,
            items=items,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            stage_code=stage_code,
            stage_name=stage_name,
            debug=self.main_window.rec_chk_debug.isChecked(),
        )
        self.main_window._recorder.start()

        self.main_window._recorder_poll_timer = QTimer(self.main_window)
        self.main_window._recorder_poll_timer.timeout.connect(self._poll_recorder_state)
        self.main_window._recorder_poll_timer.start(100)

        self.main_window.btn_rec_start.setEnabled(False)
        self.main_window.btn_rec_stop.setEnabled(True)
        self.main_window.btn_rec_save.setEnabled(False)
        self.main_window.rec_status.setText("状态: 录制器已加载，可进入作战...")

    def _poll_recorder_state(self):
        if self.main_window._recorder is None:
            return
        if self.main_window._recorder.is_stop_requested():
            self._stop_recording()
            return
        if self.main_window._recorder.is_recording() and hasattr(self.main_window._recorder, '_state'):
            if self.main_window._recorder._state == "IDLE" and self.main_window.rec_status.text().startswith("状态: 录制器已加载"):
                self.main_window.rec_status.setText("状态: 录制中")

    def _stop_recording(self):
        if hasattr(self.main_window, "_recorder_poll_timer") and self.main_window._recorder_poll_timer is not None:
            self.main_window._recorder_poll_timer.stop()
            self.main_window._recorder_poll_timer = None
        if self.main_window._recorder is None:
            return
        script = self.main_window._recorder.stop()
        self.main_window._last_recorded_script = script
        self.main_window._recorder = None
        self.main_window._recorder_capture = None

        self.main_window.btn_rec_start.setEnabled(True)
        self.main_window.btn_rec_stop.setEnabled(False)
        self.main_window.btn_rec_save.setEnabled(True)
        self.main_window.rec_status.setText(f"状态: 录制完成，共 {len(script.actions)} 个操作")

    def _save_recording(self):
        if not hasattr(self.main_window, "_last_recorded_script") or self.main_window._last_recorded_script is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "保存录制脚本", "", "JSON 文件 (*.json)"
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.main_window._last_recorded_script.model_dump(), f, ensure_ascii=False, indent=2)
            QMessageBox.information(self.main_window, "保存成功", f"已保存到:\n{path}")
        except Exception as e:
            QMessageBox.critical(self.main_window, "保存失败", str(e))

    def _rec_add_operator(self):
        name = self.main_window.rec_op_input.text().strip()
        if name:
            self.main_window.rec_op_list.addItem(name)
            self.main_window.rec_op_input.clear()
            self.main_window._save_config()

    def _rec_remove_operator(self):
        for item in self.main_window.rec_op_list.selectedItems():
            self.main_window.rec_op_list.takeItem(self.main_window.rec_op_list.row(item))
        self.main_window._save_config()

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
