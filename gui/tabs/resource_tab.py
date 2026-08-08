import os
import shutil

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox,
    QFileDialog, QGroupBox, QCheckBox, QProgressBar,
)

from core.base.paths import game_data
from core.update.levels_updater import LevelsUpdater, UpdateError
from gui.workers.levels_update_worker import LevelsUpdateWorker


class ResourceTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()
        self._update_worker: LevelsUpdateWorker | None = None

    def _project_root(self) -> str:
        return self.main_window._project_root()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---- levels.json 手动更新 ----
        manual_group = QGroupBox("levels.json 手动更新")
        manual_layout = QVBoxLayout(manual_group)
        manual_layout.addWidget(
            QLabel("选择新的 levels.json 文件，点击更新后将会覆盖 resource/game_data/levels.json")
        )

        file_layout = QHBoxLayout()
        self.main_window.resource_path = QLineEdit()
        self.main_window.resource_path.setPlaceholderText("选择 levels.json 文件...")
        file_layout.addWidget(self.main_window.resource_path)
        self.main_window.btn_resource_browse = QPushButton("浏览")
        self.main_window.btn_resource_browse.clicked.connect(self._browse_resource)
        file_layout.addWidget(self.main_window.btn_resource_browse)
        manual_layout.addLayout(file_layout)

        self.main_window.btn_update_resource = QPushButton("更新资源")
        self.main_window.btn_update_resource.clicked.connect(self._update_resource)
        manual_layout.addWidget(self.main_window.btn_update_resource)

        self.main_window.resource_status = QLabel("状态: 未更新")
        manual_layout.addWidget(self.main_window.resource_status)
        layout.addWidget(manual_group)

        # ---- levels.json 自动更新 ----
        auto_group = QGroupBox("levels.json 自动更新")
        auto_layout = QVBoxLayout(auto_group)

        self.main_window.chk_levels_auto_update = QCheckBox("启动时自动检查更新")
        self.main_window.chk_levels_auto_update.stateChanged.connect(self._on_auto_update_changed)
        auto_layout.addWidget(self.main_window.chk_levels_auto_update)

        self.main_window.chk_levels_auto_download = QCheckBox("发现更新时自动下载")
        self.main_window.chk_levels_auto_download.stateChanged.connect(self._on_auto_download_changed)
        auto_layout.addWidget(self.main_window.chk_levels_auto_download)

        url_layout = QHBoxLayout()
        url_layout.addWidget(QLabel("metadata URL:"))
        self.main_window.line_levels_metadata_url = QLineEdit()
        self.main_window.line_levels_metadata_url.setPlaceholderText(
            LevelsUpdater.DEFAULT_METADATA_URL
        )
        url_layout.addWidget(self.main_window.line_levels_metadata_url)
        auto_layout.addLayout(url_layout)

        btn_layout = QHBoxLayout()
        self.main_window.btn_levels_check_update = QPushButton("立即检查更新")
        self.main_window.btn_levels_check_update.clicked.connect(self._check_update)
        btn_layout.addWidget(self.main_window.btn_levels_check_update)
        btn_layout.addStretch()
        auto_layout.addLayout(btn_layout)

        self.main_window.progress_levels_update = QProgressBar()
        self.main_window.progress_levels_update.setRange(0, 100)
        self.main_window.progress_levels_update.setValue(0)
        self.main_window.progress_levels_update.setTextVisible(True)
        self.main_window.progress_levels_update.setFormat("%p%")
        auto_layout.addWidget(self.main_window.progress_levels_update)

        self.main_window.label_levels_update_status = QLabel("状态: 未检查")
        auto_layout.addWidget(self.main_window.label_levels_update_status)
        layout.addWidget(auto_group)

        layout.addStretch()

    def _browse_resource(self):
        path, _ = QFileDialog.getOpenFileName(self.main_window, "选择 levels.json", "", "JSON (*.json)")
        if path:
            self.main_window.resource_path.setText(path)

    def _update_resource(self):
        src = self.main_window.resource_path.text()
        if not src:
            QMessageBox.warning(self.main_window, "警告", "请先选择 levels.json 文件")
            return
        dst = str(game_data("levels.json"))
        try:
            shutil.copy2(src, dst)
            self.main_window.resource_status.setText(f"状态: 更新成功 -> {dst}")
            QMessageBox.information(self.main_window, "成功", f"已更新 levels.json\n目标: {dst}")
        except Exception as e:
            self.main_window.resource_status.setText(f"状态: 更新失败 - {e}")
            QMessageBox.critical(self.main_window, "错误", f"更新失败: {e}")

    def _on_auto_update_changed(self, state: int):
        enabled = state == Qt.CheckState.Checked.value
        self.main_window.chk_levels_auto_download.setEnabled(enabled)
        if not enabled:
            self.main_window.chk_levels_auto_download.setChecked(False)
        self.main_window._save_config()

    def _on_auto_download_changed(self, state: int):
        self.main_window._save_config()

    def _metadata_url(self) -> str:
        url = self.main_window.line_levels_metadata_url.text().strip()
        return url or LevelsUpdater.DEFAULT_METADATA_URL

    def _check_update(self):
        if self._update_worker is not None and self._update_worker.isRunning():
            QMessageBox.information(self.main_window, "提示", "正在检查更新，请稍候")
            return

        self.main_window.btn_levels_check_update.setEnabled(False)
        self.main_window.progress_levels_update.setValue(0)
        self.main_window.label_levels_update_status.setText("状态: 正在检查...")

        auto_download = self.main_window.chk_levels_auto_download.isChecked()
        self._update_worker = LevelsUpdateWorker(
            metadata_url=self._metadata_url(),
            auto_download=auto_download,
            parent=self,
        )
        self._update_worker.check_finished.connect(self._on_check_finished)
        self._update_worker.download_progress.connect(self._on_download_progress)
        self._update_worker.download_finished.connect(self._on_download_finished)
        self._update_worker.error_occurred.connect(self._on_error)
        self._update_worker.finished.connect(self._on_worker_finished)
        self._update_worker.start()

    def _on_check_finished(self, update_available: bool, info_dict: dict | None):
        if not update_available:
            self.main_window.label_levels_update_status.setText("状态: 已是最新版本")
            return

        size = info_dict.get("size", 0)
        updated_at = info_dict.get("updated_at", "")
        self.main_window.label_levels_update_status.setText(
            f"发现新版本 (size: {size}, updated_at: {updated_at})"
        )

    def _on_download_progress(self, current: int, total: int):
        if total > 0:
            percent = int(current / total * 100)
            self.main_window.progress_levels_update.setValue(percent)
        self.main_window.label_levels_update_status.setText(
            f"下载中: {current / 1024 / 1024:.1f} / {total / 1024 / 1024:.1f} MB"
        )

    def _on_download_finished(self, success: bool, message: str):
        if success:
            self.main_window.label_levels_update_status.setText(f"状态: {message}")
            self.main_window.resource_status.setText("状态: 已通过自动更新同步")
            QMessageBox.information(self.main_window, "更新成功", message)
        else:
            self.main_window.label_levels_update_status.setText(f"状态: 更新失败 - {message}")
            QMessageBox.critical(self.main_window, "更新失败", message)

    def _on_error(self, message: str):
        self.main_window.label_levels_update_status.setText(f"状态: 检查失败 - {message}")
        QMessageBox.critical(self.main_window, "检查失败", message)

    def _on_worker_finished(self):
        self.main_window.btn_levels_check_update.setEnabled(True)
        self._update_worker = None

    def trigger_auto_check(self):
        """由 MainWindow 在启动时调用，仅在自动检查启用时执行。"""
        if not self.main_window.chk_levels_auto_update.isChecked():
            return
        self._check_update()
