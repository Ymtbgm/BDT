import json
import time

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QSpinBox, QCheckBox,
    QGroupBox, QListWidget, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QMessageBox, QFileDialog, QComboBox,
    QApplication, QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer

import action
from core.capture import WindowCapture
from core.ocr_engine import OCREngine
from core.recorder import ActionRecorder
from core.region_state_timer import RegionStateTimer
from gui.info_collection_overlay import InfoCollectionOverlay
from gui.widgets.checked_combo_box import CheckedComboBox
from models.script_schema import ItemInfo


class RecorderTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 参数区域：每行一个 QHBoxLayout，输入框紧跟标签，不居中展开
        param_group = QGroupBox("录制参数")
        param_layout = QVBoxLayout()

        row = QHBoxLayout()
        row.addWidget(QLabel("关卡代号:"))
        self.main_window.rec_stage_code = QLineEdit()
        self.main_window.rec_stage_code.setPlaceholderText("如 1-7")
        self.main_window.rec_stage_code.setMaximumWidth(80)
        row.addWidget(self.main_window.rec_stage_code)
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
        row.addWidget(QLabel("debug选项:"))
        self.main_window.combo_rec_debug = CheckedComboBox("未选择")
        self.main_window.combo_rec_debug.add_item("录制器状态机", "recorder")
        self.main_window.combo_rec_debug.add_item("费用条检测", "cost_bar")
        self.main_window.combo_rec_debug.add_item("离线识别", "resolver")
        self.main_window.combo_rec_debug.add_item("调试截图", "screenshot")
        self.main_window.combo_rec_debug.setToolTip(
            "勾选需要打印日志或保存截图的调试项，可多选。"
        )
        self.main_window.combo_rec_debug.setMaximumWidth(200)
        row.addWidget(self.main_window.combo_rec_debug)
        row.addStretch()
        param_layout.addLayout(row)

        # 干员列表 + 道具列表并排
        op_group = QGroupBox("干员列表")
        op_layout = QVBoxLayout(op_group)
        self.main_window.rec_op_table = QTableWidget()
        self.main_window.rec_op_table.setColumnCount(2)
        self.main_window.rec_op_table.setHorizontalHeaderLabels(["干员名", "费用"])
        self.main_window.rec_op_table.setMinimumHeight(100)
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
        self.main_window.rec_op_input = QLineEdit()
        self.main_window.rec_op_input.setPlaceholderText("输入干员名...")
        self.main_window.rec_op_cost_input = QSpinBox()
        self.main_window.rec_op_cost_input.setRange(0, 999)
        self.main_window.rec_op_cost_input.setValue(0)
        self.main_window.rec_op_cost_input.setSpecialValueText("未设置")
        self.main_window.rec_op_add_btn = QPushButton("添加")
        self.main_window.rec_op_remove_btn = QPushButton("删除")
        self.main_window.rec_op_up_btn = QPushButton("上移")
        self.main_window.rec_op_down_btn = QPushButton("下移")
        op_input_layout.addWidget(self.main_window.rec_op_input)
        op_input_layout.addWidget(self.main_window.rec_op_cost_input)
        op_input_layout.addWidget(self.main_window.rec_op_add_btn)
        op_input_layout.addWidget(self.main_window.rec_op_remove_btn)
        op_input_layout.addWidget(self.main_window.rec_op_up_btn)
        op_input_layout.addWidget(self.main_window.rec_op_down_btn)
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
        self.main_window.rec_item_up_btn = QPushButton("上移")
        self.main_window.rec_item_down_btn = QPushButton("下移")
        item_input_layout.addWidget(self.main_window.rec_item_input)
        item_input_layout.addWidget(self.main_window.rec_item_charges_input)
        item_input_layout.addWidget(self.main_window.rec_item_add_btn)
        item_input_layout.addWidget(self.main_window.rec_item_remove_btn)
        item_input_layout.addWidget(self.main_window.rec_item_up_btn)
        item_input_layout.addWidget(self.main_window.rec_item_down_btn)
        item_layout.addLayout(item_input_layout)

        list_layout = QHBoxLayout()
        list_layout.addWidget(op_group, 1)
        list_layout.addWidget(item_group, 1)
        param_layout.addLayout(list_layout)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

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
        self.main_window.combo_rec_debug.item_changed.connect(self.main_window._save_config)
        self.main_window.rec_chk_support_op.stateChanged.connect(self.main_window._save_config)

        # 使用说明
        guide_label = QLabel()
        guide_label.setWordWrap(True)
        guide_label.setTextFormat(Qt.TextFormat.RichText)
        guide_label.setText(
            "<h3>使用说明</h3>"
            "<ul>"
            "<li>在编队界面点击开始录制，引导显示 加载完毕，请进入作战... 即可进入作战。</li>"
            "<li>进入作战后检测到费用条启动即开始记录。</li>"
            "<li>干员被击退前请主动撤退，否则也会错位。</li>"
            "<li>F10停止录制。</li>"
            "<li>录制结束后会自动离线识别并生成脚本，若出现 __unknown__ / __item__ 可以手动修正，这是因为该单位未被部署，不修正不会影响脚本正常执行。</li>"
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

        # 与脚本执行 Tab 保持一致的勾选框样式：选中时用绿色填充，不显示默认对勾
        checkbox_green_fill_style = (
            "QCheckBox::indicator { width: 13px; height: 13px; border: 1px solid #1a1a1a; border-radius: 3px; image: none; }"
            "QCheckBox::indicator:checked { background-color: #4CAF50; }"
        )
        for checkbox in self.findChildren(QCheckBox):
            checkbox.setStyleSheet(checkbox_green_fill_style)

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

    def _start_recording(self):
        initial_operator_count = self.main_window.rec_initial_operator_count.value()
        if initial_operator_count <= 0:
            QMessageBox.warning(self.main_window, "参数错误", "请在“初始干员数量”中填写编队携带的干员数量（至少 1）。")
            return

        stage_code = self.main_window.rec_stage_code.text().strip()
        if not stage_code:
            QMessageBox.warning(self.main_window, "参数错误", "请填写关卡代号（如 1-7）。")
            return

        try:
            self.main_window._recorder_capture = WindowCapture(backend="mss")
        except Exception as e:
            QMessageBox.critical(self.main_window, "错误", f"窗口捕获初始化失败:\n{e}")
            return

        initial_item_count = self.main_window.rec_initial_item_count.value()
        support_count = 1 if self.main_window.rec_chk_support_op.isChecked() else 0
        debug_keys = set(self.main_window.combo_rec_debug.checked_data())
        rec_debug = "recorder" in debug_keys
        rec_debug_cost_bar = "cost_bar" in debug_keys
        rec_debug_resolver = "resolver" in debug_keys
        rec_debug_screenshot = "screenshot" in debug_keys

        self._normal_overlay_started = False
        self._recorder_overlay_timer_mode = False
        self._recording_overlay_switch_at = 0.0

        overlay = InfoCollectionOverlay(debug=rec_debug)
        overlay.set_phase("正在初始化 OCR...")
        overlay.show()
        QApplication.processEvents()

        ocr = None
        try:
            ocr = OCREngine(engine=None, debug=rec_debug, use_gpu=True)
        except Exception as e:
            overlay.close_overlay()
            QMessageBox.critical(self.main_window, "错误", f"OCR 初始化失败:\n{e}")
            return

        self.main_window.showMinimized()
        overlay.set_phase("请保持编队界面...")
        QApplication.processEvents()

        def _resolver_log(line: str):
            self.main_window.log_text.append(line)
            QApplication.processEvents()

        self.main_window._recorder = ActionRecorder(
            capture=self.main_window._recorder_capture,
            timer=self.main_window._region_timer,
            stage_code=stage_code,
            debug=rec_debug,
            debug_cost_bar=rec_debug_cost_bar,
            debug_resolver=rec_debug_resolver,
            debug_screenshot=rec_debug_screenshot,
            initial_operator_count=initial_operator_count,
            initial_item_count=initial_item_count,
            support_count=support_count,
            pause_key=action.pause_key(),
            matchstick_hotkeys=self.main_window.timer_tab._build_matchstick_hotkeys() or None,
            cost_bar_calibration_name=self.main_window.combo_timer_cost_tag.currentData() or None,
            ocr=ocr,
            resolver_log_callback=_resolver_log if rec_debug_resolver else None,
        )
        self.main_window._recorder_overlay = overlay
        self.main_window._recorder.start()

        self.main_window._recorder_poll_timer = QTimer(self.main_window)
        self.main_window._recorder_poll_timer.timeout.connect(self._poll_recorder_state)
        self.main_window._recorder_poll_timer.start(33)

        self.main_window.btn_rec_start.setEnabled(False)
        self.main_window.btn_rec_stop.setEnabled(True)
        self.main_window.btn_rec_save.setEnabled(False)
        self.main_window.rec_status.setText("加载完毕，请进入作战...")

    def _poll_recorder_state(self):
        if self.main_window._recorder is None:
            return
        state = getattr(self.main_window._recorder, "_state", "")
        if self.main_window._recorder.is_stop_requested():
            self._stop_recording()
            return
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        debug = "recorder" in set(self.main_window.combo_rec_debug.checked_data())
        if debug:
            print(f"[recorder_tab poll] state={state} started={getattr(self, '_normal_overlay_started', False)} timer_mode={getattr(self, '_recorder_overlay_timer_mode', False)} switch_at={getattr(self, '_recording_overlay_switch_at', 0):.3f} now={time.time():.3f}")
        if overlay is not None:
            if state == "WAITING_FOR_START":
                if getattr(self.main_window._recorder, "is_squad_capture_done", lambda: False)():
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
                    self._recording_overlay_switch_at = time.time() + 5
                elif time.time() >= self._recording_overlay_switch_at:
                    if debug:
                        print("[recorder_tab poll] switch to timer")
                    self._switch_recorder_overlay_to_timer()
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
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        print(f"[_switch_recorder_overlay_to_timer] overlay={overlay}")
        if overlay is None:
            return
        self._recorder_overlay_timer_mode = True
        timer = self.main_window._region_timer
        if timer is None and self.main_window._recorder is not None:
            timer = self.main_window._recorder.timer
        print(f"[_switch_recorder_overlay_to_timer] timer={timer} started={timer.is_started() if timer else None}")
        if timer is not None and timer.is_started():
            elapsed = timer.get_elapsed_ms()
            s, f = self.main_window._ms_to_sf_for_timer(elapsed)
            print(f"[_switch_recorder_overlay_to_timer] elapsed={elapsed:.1f} s={s} f={f}")
            overlay.set_time(
                s, f, elapsed,
                getattr(timer, "_rate", 1.0),
                timer.is_manual_paused(),
            )
        else:
            print("[_switch_recorder_overlay_to_timer] timer not started, show 0")
            overlay.set_time(0, 0)

    def _update_recorder_overlay_time(self, overlay):
        """刷新录制浮窗的计时显示。"""
        timer = self.main_window._region_timer
        if timer is None and self.main_window._recorder is not None:
            timer = self.main_window._recorder.timer
        if timer is None:
            print("[_update_recorder_overlay_time] timer is None")
            return
        elapsed = timer.get_elapsed_ms()
        s, f = self.main_window._ms_to_sf_for_timer(elapsed)
        print(f"[_update_recorder_overlay_time] elapsed={elapsed:.1f} s={s} f={f}")
        overlay.set_time(
            s, f, elapsed,
            getattr(timer, "_rate", 1.0),
            timer.is_manual_paused(),
        )

    def _stop_recording(self):
        self._normal_overlay_started = False
        self._recorder_overlay_timer_mode = False
        self._recording_overlay_switch_at = 0.0
        if hasattr(self.main_window, "_recorder_poll_timer") and self.main_window._recorder_poll_timer is not None:
            self.main_window._recorder_poll_timer.stop()
            self.main_window._recorder_poll_timer = None
        overlay = getattr(self.main_window, "_recorder_overlay", None)
        if overlay is not None:
            overlay.close_overlay()
            self.main_window._recorder_overlay = None
        script = None
        if self.main_window._recorder is not None:
            script = self.main_window._recorder.stop()
            self.main_window._last_recorded_script = script
            self.main_window._recorder = None
        self.main_window._recorder_capture = None
        self.main_window.showNormal()

        self.main_window.btn_rec_start.setEnabled(True)
        self.main_window.btn_rec_stop.setEnabled(False)
        self.main_window.btn_rec_save.setEnabled(True)
        recorded = getattr(self.main_window, "_last_recorded_script", None)
        action_count = len(recorded.actions) if recorded is not None else 0
        self.main_window.rec_status.setText(f"状态: 录制完成，共 {action_count} 个操作")

        if recorded is not None:
            self._populate_recorder_tables(recorded)

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
        path, _ = QFileDialog.getSaveFileName(
            self.main_window, "保存录制脚本", "", "JSON 文件 (*.json)"
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

