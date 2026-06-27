import os
import shutil

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox, QFileDialog,
)


class ResourceTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("levels.json 资源更新"))
        layout.addWidget(
            QLabel("选择新的 levels.json 文件，点击更新后将会覆盖 core/resource/levels.json")
        )

        file_layout = QHBoxLayout()
        self.main_window.resource_path = QLineEdit()
        self.main_window.resource_path.setPlaceholderText("选择 levels.json 文件...")
        file_layout.addWidget(self.main_window.resource_path)
        self.main_window.btn_resource_browse = QPushButton("浏览")
        self.main_window.btn_resource_browse.clicked.connect(self._browse_resource)
        file_layout.addWidget(self.main_window.btn_resource_browse)
        layout.addLayout(file_layout)

        self.main_window.btn_update_resource = QPushButton("更新资源")
        self.main_window.btn_update_resource.clicked.connect(self._update_resource)
        layout.addWidget(self.main_window.btn_update_resource)

        self.main_window.resource_status = QLabel("状态: 未更新")
        layout.addWidget(self.main_window.resource_status)

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
        dst = os.path.join(self.main_window._project_root(), "core", "resource", "levels.json")
        try:
            shutil.copy2(src, dst)
            self.main_window.resource_status.setText(f"状态: 更新成功 -> {dst}")
            QMessageBox.information(self.main_window, "成功", f"已更新 levels.json\n目标: {dst}")
        except Exception as e:
            self.main_window.resource_status.setText(f"状态: 更新失败 - {e}")
            QMessageBox.critical(self.main_window, "错误", f"更新失败: {e}")
