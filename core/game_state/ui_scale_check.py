"""明日方舟游戏内 UI 比例检测。

从 Windows 注册表读取 Hypergryph/Arknights 的 common_setting，
解析其中的 uiScaler 字段，若不为 0.0（对应游戏内 90% UI）则提示用户调整。
"""

import json
import re
import winreg
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import QMessageBox, QCheckBox


_HKCU = winreg.HKEY_CURRENT_USER
_BASE_KEY = r"Software\Hypergryph\Arknights"
_VALUE_NAME = "common_setting_h2012961537"
_VALUE_RE = re.compile(r"^common_setting_h\d+$")
_TARGET_SCALE = 0.0
_TOLERANCE = 0.001


_KNOWN_SUBKEYS = [
    r"18327005#{0}_personal_setting_h3196625204",
]


def _access_flags():
    """返回要尝试的注册表访问标志组合（含 WOW64 视图）。"""
    return [
        ("default", winreg.KEY_READ),
        ("wow64_64", winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
        ("wow64_32", winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
    ]


def _open_key_robust(key_path: str):
    """尝试多种访问标志打开注册表键，返回 (key_handle, access_name) 或 (None, None)。"""
    for access_name, access in _access_flags():
        try:
            key = winreg.OpenKey(_HKCU, key_path, 0, access)
            return key, access_name
        except Exception:
            continue
    return None, None


def _list_subkeys() -> list[str]:
    """返回 Arknights 下的所有子键名，支持 WOW64 视图。"""
    names = []
    base_key, _ = _open_key_robust(_BASE_KEY)
    if base_key is None:
        return names
    try:
        with base_key:
            idx = 0
            while True:
                try:
                    names.append(winreg.EnumKey(base_key, idx))
                    idx += 1
                except OSError:
                    break
    except Exception:
        pass
    return names


def _parse_ui_scaler(text: str) -> Optional[float]:
    """从 common_setting JSON 文本中解析 uiScaler。"""
    try:
        setting = json.loads(text)
        value = setting.get("uiScaler")
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _try_read_scaler_from_key(key_path: str, debug: bool = False) -> Optional[float]:
    """从指定键路径读取 uiScaler，失败返回 None。

    先尝试固定值名，找不到时再按正则 common_setting_h\\d+ 枚举匹配。
    """
    key, access_name = _open_key_robust(key_path)
    if key is None:
        if debug:
            print(f"[ui_scale_check] 无法打开 {key_path}")
        return None

    def _read(name: str) -> Optional[float]:
        raw, reg_type = winreg.QueryValueEx(key, name)
        if reg_type == winreg.REG_BINARY:
            data = bytes(raw).rstrip(b"\x00")
        elif reg_type == winreg.REG_SZ:
            data = str(raw).encode("utf-8")
        else:
            if debug:
                print(f"[ui_scale_check] {key_path} ({access_name}) 值 {name} 不支持的类型: {reg_type}")
            return None

        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        text = data.decode("utf-8", errors="replace")
        scaler = _parse_ui_scaler(text)
        if debug and scaler is not None:
            print(f"[ui_scale_check] 从 {key_path} ({access_name}) [{name}] 读取到 uiScaler={scaler}")
        return scaler

    try:
        with key:
            # 优先固定值名
            try:
                scaler = _read(_VALUE_NAME)
                if scaler is not None:
                    return scaler
            except FileNotFoundError:
                pass

            # 兜底：枚举值名并按正则匹配
            idx = 0
            while True:
                try:
                    name, _, _ = winreg.EnumValue(key, idx)
                    if _VALUE_RE.match(name):
                        try:
                            scaler = _read(name)
                            if scaler is not None:
                                return scaler
                        except Exception:
                            pass
                    idx += 1
                except OSError:
                    break
    except Exception as e:
        if debug:
            print(f"[ui_scale_check] 读取 {key_path} ({access_name}) 失败: {e}")
    return None


def get_arknights_ui_scaler(debug: bool = False) -> Optional[float]:
    """读取注册表并返回 uiScaler 数值；读取失败返回 None。"""
    # 1. 先尝试基础键本身
    scaler = _try_read_scaler_from_key(_BASE_KEY, debug=debug)
    if scaler is not None:
        return scaler

    # 2. 枚举子键尝试
    for subkey_name in _list_subkeys():
        scaler = _try_read_scaler_from_key(f"{_BASE_KEY}\\{subkey_name}", debug=debug)
        if scaler is not None:
            return scaler

    # 3. 兜底：尝试已知的完整子键路径
    for known in _KNOWN_SUBKEYS:
        scaler = _try_read_scaler_from_key(f"{_BASE_KEY}\\{known}", debug=debug)
        if scaler is not None:
            return scaler

    return None


def check_ui_scale(parent=None, config: Optional[dict] = None) -> bool:
    """检查 UI 比例，非 90% 时弹窗提示。

    返回 True 表示继续启动（比例正确、用户忽略、或读取失败）；
    返回 False 表示用户选择退出。
    """
    if config and config.get("ui_scale_check_disabled"):
        return True

    scaler = get_arknights_ui_scaler()
    if scaler is None:
        return True

    if abs(scaler - _TARGET_SCALE) < _TOLERANCE:
        return True

    msg = (
        f"检测到明日方舟 UI 比例为 {int(scaler * 100)}%，本工具当前仅适配 90% UI 比例。\n"
        "请在游戏设置中将“UI 大小”调整为 90%，否则识别和点击可能出现偏差。\n\n"
        "你可以先忽略此提示继续启动，或退出调整后再使用。"
    )

    box = QMessageBox(parent)
    box.setWindowTitle("UI 比例提示")
    box.setText(msg)
    box.setIcon(QMessageBox.Icon.Warning)
    btn_ignore = box.addButton("忽略并继续", QMessageBox.ButtonRole.AcceptRole)
    btn_exit = box.addButton("退出", QMessageBox.ButtonRole.RejectRole)
    box.setDefaultButton(btn_exit)

    cb = QCheckBox("不再提示")
    box.setCheckBox(cb)

    box.exec()
    clicked = box.clickedButton()
    dont_ask = cb.isChecked()

    if clicked == btn_exit:
        return False

    if dont_ask and config is not None:
        config["ui_scale_check_disabled"] = True

    return True
