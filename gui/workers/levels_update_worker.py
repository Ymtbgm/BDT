"""levels.json 自动更新后台工作线程。"""

from PyQt6.QtCore import QThread, pyqtSignal

from core.update.levels_updater import LevelsUpdater, UpdateInfo, UpdateError


class LevelsUpdateWorker(QThread):
    """在后台执行 levels.json 检查/下载，避免阻塞 GUI。"""

    check_finished = pyqtSignal(bool, object)  # (update_available, info_dict|None)
    download_progress = pyqtSignal(int, int)  # (current, total)
    download_finished = pyqtSignal(bool, str)  # (success, message)
    error_occurred = pyqtSignal(str)

    def __init__(self, metadata_url: str, auto_download: bool = True, parent=None):
        super().__init__(parent)
        self._metadata_url = metadata_url
        self._auto_download = auto_download
        self._updater = LevelsUpdater(metadata_url=metadata_url)
        self._info: UpdateInfo | None = None

    def run(self):
        try:
            self._info = self._updater.check_update()
        except UpdateError as e:
            self.error_occurred.emit(str(e))
            return

        if self._info is None:
            self.check_finished.emit(False, None)
            return

        info_dict = {
            "filename": self._info.filename,
            "size": self._info.size,
            "sha256": self._info.sha256,
            "updated_at": self._info.updated_at.isoformat() if self._info.updated_at else "",
            "source": self._info.source,
            "url": self._info.url,
            "metadata_url": self._info.metadata_url,
        }
        self.check_finished.emit(True, info_dict)

        if not self._auto_download:
            return

        try:
            temp_path = self._updater.download(self._info, progress_callback=self._on_progress)
            self._updater.install(self._info, temp_path)
            self.download_finished.emit(
                True,
                f"更新完成: {self._info.filename} ({self._info.size} bytes)",
            )
        except UpdateError as e:
            self.download_finished.emit(False, str(e))

    def _on_progress(self, current: int, total: int):
        self.download_progress.emit(current, total)

    def set_auto_download(self, enabled: bool):
        self._auto_download = enabled
