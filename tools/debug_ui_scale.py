"""调试 UI 比例注册表读取。

不依赖 PyQt6，直接在控制台打印注册表遍历和解析结果。
运行方式：
    python tools/debug_ui_scale.py
"""

import json
import re
import sys
import winreg

_HKCU = winreg.HKEY_CURRENT_USER
_BASE_KEY = r"Software\Hypergryph\Arknights"
_VALUE_NAME = "common_setting_h2012961537"
_VALUE_RE = re.compile(r"^common_setting_h\d+$")
_KNOWN_SUBKEY = r"18327005#{0}_personal_setting_h3196625204"


def _try_open(key_path, access=winreg.KEY_READ):
    try:
        return winreg.OpenKey(_HKCU, key_path, 0, access)
    except Exception as e:
        return None, e


def _read_value(key, value_name: str):
    raw, reg_type = winreg.QueryValueEx(key, value_name)
    print(f"  匹配值名: {value_name}")
    print(f"  目标值类型: {reg_type} (REG_BINARY={winreg.REG_BINARY}, REG_SZ={winreg.REG_SZ})")
    print(f"  原始字节/字符串长度: {len(raw)}")

    if reg_type == winreg.REG_BINARY:
        data = bytes(raw).rstrip(b"\x00")
    elif reg_type == winreg.REG_SZ:
        data = str(raw).encode("utf-8")
    else:
        print(f"  不支持的类型，跳过")
        return None

    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    print(f"  前 80 字节(hex): {data[:80].hex()}")
    text = data.decode("utf-8", errors="replace")
    print(f"  解码后前 200 字符: {text[:200]}")
    setting = json.loads(text)
    ui_scaler = setting.get("uiScaler")
    print(f"  uiScaler = {ui_scaler!r} (类型: {type(ui_scaler).__name__})")
    return ui_scaler


def _find_common_setting_value_name(key):
    """先返回精确值名，找不到则返回第一个匹配 common_setting_h\\d+ 的值名。"""
    names = []
    idx = 0
    while True:
        try:
            vname, _, _ = winreg.EnumValue(key, idx)
            names.append(vname)
            idx += 1
        except OSError:
            break

    if _VALUE_NAME in names:
        return _VALUE_NAME
    for name in names:
        if _VALUE_RE.match(name):
            return name
    return None


def _inspect_key(key_path):
    print(f"\n尝试打开: {_HKCU_NAME}\\{key_path}")
    for access_name, access in (
        ("默认", winreg.KEY_READ),
        ("WOW64_64KEY", winreg.KEY_READ | winreg.KEY_WOW64_64KEY),
        ("WOW64_32KEY", winreg.KEY_READ | winreg.KEY_WOW64_32KEY),
    ):
        try:
            with winreg.OpenKey(_HKCU, key_path, 0, access) as key:
                print(f"  [{access_name}] 成功")
                # 列值
                print("  该键下的值名:")
                vidx = 0
                while True:
                    try:
                        vname, _, vtype = winreg.EnumValue(key, vidx)
                        print(f"    [{vidx}] {vname} (类型={vtype})")
                        vidx += 1
                    except OSError:
                        break
                if vidx == 0:
                    print("    (无值)")
                # 尝试读取目标值
                value_name = _find_common_setting_value_name(key)
                if value_name is None:
                    print(f"  该键下没有匹配 common_setting_h\\d+ 的值")
                else:
                    try:
                        return _read_value(key, value_name)
                    except Exception as e:
                        print(f"  读取 {value_name} 失败: {type(e).__name__}: {e}")
        except Exception as e:
            print(f"  [{access_name}] 失败: {type(e).__name__}: {e}")
    return None


_HKCU_NAME = "HKEY_CURRENT_USER"


def main():
    print(f"Python 位数: {sys.maxsize.bit_length() + 1}")
    print(f"基础键: {_BASE_KEY}")

    # 1. 检查基础键本身
    _inspect_key(_BASE_KEY)

    # 2. 枚举子键
    print(f"\n枚举 {_BASE_KEY} 子键:")
    try:
        with winreg.OpenKey(_HKCU, _BASE_KEY) as base:
            idx = 0
            while True:
                try:
                    name = winreg.EnumKey(base, idx)
                    print(f"  [{idx}] {name}")
                    idx += 1
                except OSError:
                    break
            if idx == 0:
                print("  (无子键)")
    except Exception as e:
        print(f"  枚举失败: {e}")

    # 3. 直接尝试已知子键
    full_path = f"{_BASE_KEY}\\{_KNOWN_SUBKEY}"
    _inspect_key(full_path)


if __name__ == "__main__":
    main()
