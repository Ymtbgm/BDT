"""levels.json 自动更新器。

从远程 metadata.json 检查更新，下载 levels.json 到临时文件，校验 sha256 后原子替换本地文件。
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import requests

from core.map.tile_pos import invalidate_levels_cache
from core.base.paths import game_data


@dataclass(frozen=True)
class UpdateInfo:
    """远程 metadata 解析后的信息。"""

    filename: str
    size: int
    sha256: str
    updated_at: Optional[datetime]
    source: str
    url: str
    metadata_url: str


class UpdateError(Exception):
    """更新过程中出现的错误。"""

    pass


class LevelsUpdater:
    """负责检查、下载并安装 levels.json 更新。"""

    DEFAULT_METADATA_URL = "https://levelcunchu1.oss-cn-beijing.aliyuncs.com/levels/metadata.json"
    REQUEST_TIMEOUT = 30
    DOWNLOAD_TIMEOUT = 300

    def __init__(
        self,
        metadata_url: Optional[str] = None,
        target_path: Optional[Path] = None,
    ):
        self.metadata_url = metadata_url or self.DEFAULT_METADATA_URL
        self.target_path = target_path if target_path is not None else game_data("levels.json")
        self.target_path.parent.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({
            "Accept": "application/json",
            "Accept-Encoding": "gzip, deflate",
        })

    def check_update(self) -> Optional[UpdateInfo]:
        """拉取远程 metadata，与本地文件比对，返回是否需要更新。

        若本地文件不存在、metadata 无法解析、或网络失败则抛出 UpdateError。
        若远程文件与本地一致则返回 None。
        """
        try:
            resp = self._session.get(self.metadata_url, timeout=self.REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            raise UpdateError(f"无法获取远程 metadata: {e}") from e
        except json.JSONDecodeError as e:
            raise UpdateError(f"metadata 不是合法 JSON: {e}") from e

        info = self._parse_metadata(data)

        local_sha256 = self._local_sha256()
        if local_sha256 and local_sha256.lower() == info.sha256.lower():
            return None

        return info

    def download(
        self,
        info: UpdateInfo,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """下载远程 levels.json 到临时文件，返回临时文件路径。

        下载过程中可选通过 progress_callback(current, total) 报告进度。
        """
        try:
            resp = self._session.get(info.url, stream=True, timeout=self.DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
        except requests.RequestException as e:
            raise UpdateError(f"下载失败: {e}") from e

        total = int(resp.headers.get("Content-Length", info.size))
        tmp_fd, tmp_path = tempfile.mkstemp(
            prefix="levels.json.tmp.",
            dir=str(self.target_path.parent),
        )
        downloaded = 0
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_callback:
                        progress_callback(downloaded, total)
        except Exception as e:
            Path(tmp_path).unlink(missing_ok=True)
            raise UpdateError(f"写入临时文件失败: {e}") from e

        return Path(tmp_path)

    def install(self, info: UpdateInfo, temp_path: Path) -> None:
        """校验临时文件 sha256 并原子替换目标文件。"""
        if not temp_path.exists():
            raise UpdateError(f"临时文件不存在: {temp_path}")

        actual_sha256 = self._file_sha256(temp_path)
        if actual_sha256.lower() != info.sha256.lower():
            temp_path.unlink(missing_ok=True)
            raise UpdateError(
                f"校验失败: 期望 sha256 {info.sha256}, 实际 {actual_sha256}"
            )

        try:
            os.replace(str(temp_path), str(self.target_path))
        except OSError as e:
            temp_path.unlink(missing_ok=True)
            raise UpdateError(f"替换目标文件失败: {e}") from e

        # 替换成功后清空 tile_pos 的缓存，使新数据立即生效
        invalidate_levels_cache()

    def update(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> UpdateInfo:
        """一键检查并下载更新。若无需更新则返回 None，否则返回更新信息。"""
        info = self.check_update()
        if info is None:
            return None
        temp_path = self.download(info, progress_callback=progress_callback)
        self.install(info, temp_path)
        return info

    def _parse_metadata(self, data: dict) -> UpdateInfo:
        """解析 metadata 字典。"""
        try:
            updated_at_str = data.get("updated_at")
            updated_at = None
            if updated_at_str:
                updated_at = datetime.fromisoformat(updated_at_str.replace("Z", "+00:00"))

            return UpdateInfo(
                filename=str(data.get("filename", "levels.json")),
                size=int(data["size"]),
                sha256=str(data["sha256"]),
                updated_at=updated_at,
                source=str(data.get("source", "")),
                url=str(data.get("url", "")),
                metadata_url=str(data.get("metadata_url", self.metadata_url)),
            )
        except (KeyError, ValueError, TypeError) as e:
            raise UpdateError(f"metadata 字段不完整或格式错误: {e}") from e

    def _local_sha256(self) -> Optional[str]:
        """计算本地目标文件的 sha256，文件不存在时返回 None。"""
        if not self.target_path.exists():
            return None
        return self._file_sha256(self.target_path)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        """计算文件 sha256。"""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
