import pydirectinput
import time
from pynput.keyboard import Listener

# --- 核心设置 ---
pydirectinput.PAUSE = 0
interval_time = 0.166

# 可配置键位（默认与原版一致）
_KEY_PAUSE = "p"
_KEY_SKILL = "e"
_KEY_RETREAT = "q"

# 划火柴热键配置（可在 GUI 中修改）
_MATCHSTICK_HOTKEYS = {
    "select_operator": "r",
    "pass_166ms": "space",
    "pass_50ms": "f",
}
_MATCHSTICK_ENABLED = {
    "select_operator": False,
    "pass_166ms": False,
    "pass_50ms": False,
}
_keyboard_listener = None


def configure_keys(pause: str = None, skill: str = None, retreat: str = None):
    """配置游戏内快捷键，适配不同用户的键位设置。"""
    global _KEY_PAUSE, _KEY_SKILL, _KEY_RETREAT
    if pause:
        _KEY_PAUSE = pause.lower()
    if skill:
        _KEY_SKILL = skill.lower()
    if retreat:
        _KEY_RETREAT = retreat.lower()


def configure_matchstick(hotkeys: dict = None, enabled: dict = None):
    """配置划火柴全局热键及其启用状态。"""
    global _MATCHSTICK_HOTKEYS, _MATCHSTICK_ENABLED
    if hotkeys:
        _MATCHSTICK_HOTKEYS.update(hotkeys)
    if enabled:
        _MATCHSTICK_ENABLED.update(enabled)


def get_matchstick_config():
    """返回当前划火柴热键配置的副本。"""
    return {
        "hotkeys": _MATCHSTICK_HOTKEYS.copy(),
        "enabled": _MATCHSTICK_ENABLED.copy(),
    }


def pause_key() -> str:
    return _KEY_PAUSE


def skill_key() -> str:
    return _KEY_SKILL


def retreat_key() -> str:
    return _KEY_RETREAT


def _notify_timer_shield(duration_ms: float = 500.0):
    """通知主进程计时器进入划火柴保护期（通过 stdout 协议）。"""
    print(f"__TIMER_SHIELD__:{duration_ms}")
    time.sleep(0.02)  # 等待主进程接收并生效


def press_key(key: str):
    pydirectinput.keyDown(key)
    time.sleep(0.05)
    pydirectinput.keyUp(key)


def pause():
    """按下暂停键（默认 P）暂停/恢复游戏。"""
    pydirectinput.press(_KEY_PAUSE)


def select_at(x: int, y: int):
    """普通左键点击，用于在暂停状态下点击 UI（如部署栏索引）。"""
    pydirectinput.moveTo(x, y)
    pydirectinput.click(button='left')


def select_operator_matchstick(x: int, y: int):
    """划火柴选中场上干员：移动到目标位置后执行 暂停键+左键+ESC（保持暂停）。"""
    pydirectinput.moveTo(x, y)
    p_and_left_click()


def deploy_at(from_x: int, from_y: int, to_x: int, to_y: int, direction: str = None, window_w: int = 2560, window_h: int = 1600):
    """暂停状态下从干员栏拖拽部署到目标格子，并选择朝向。"""
    original = pydirectinput.position()
    pydirectinput.moveTo(from_x, from_y)
    pydirectinput.mouseDown()
    time.sleep(0.5)
    pydirectinput.moveTo(to_x, to_y)
    time.sleep(0.5)  # 等待 side 视角切换动画完成，避免松手时格子还在漂移
    pydirectinput.mouseUp()
    time.sleep(1.0)  # 等待部署 UI 弹出（方向选择圈）

    # 部署完成后选择朝向：向指定方向再拖拽一小段
    # 偏移量使用窗口短边的 1/8，确保在不同分辨率下比例一致
    if direction:
        offset = min(window_w, window_h) // 8
        dir_map = {
            "up": (0, -offset),
            "down": (0, offset),
            "left": (-offset, 0),
            "right": (offset, 0),
        }
        dx, dy = dir_map.get(direction, (0, 0))
        if dx or dy:
            pydirectinput.moveTo(to_x, to_y)
            pydirectinput.mouseDown()
            time.sleep(0.05)
            pydirectinput.moveTo(to_x + dx, to_y + dy)
            time.sleep(0.05)
            pydirectinput.mouseUp()

    pydirectinput.moveTo(original[0], original[1])


def retreat_at(x: int, y: int):
    """划火柴撤退：选中后按撤退键（默认 Q）。"""
    pydirectinput.moveTo(x, y)
    p_and_left_click()
    pydirectinput.press(_KEY_RETREAT)


def skill_at(x: int, y: int):
    """划火柴释放技能：选中后按技能键（默认 E）。"""
    pydirectinput.moveTo(x, y)
    p_and_left_click()
    pydirectinput.press(_KEY_SKILL)


def p_and_esc_click():
    _notify_timer_shield(400)
    pydirectinput.keyDown(_KEY_PAUSE)
    time.sleep(interval_time)
    pydirectinput.keyDown('esc')
    time.sleep(0.1)
    pydirectinput.keyUp('esc')
    pydirectinput.keyUp(_KEY_PAUSE)


def p_and_esc_click_short():
    """划火柴短停顿版：暂停 50ms 后接 ESC。"""
    _notify_timer_shield(400)
    pydirectinput.keyDown(_KEY_PAUSE)
    time.sleep(0.05)
    pydirectinput.keyDown('esc')
    time.sleep(0.1)
    pydirectinput.keyUp('esc')
    pydirectinput.keyUp(_KEY_PAUSE)


def p_and_left_click():
    """划火柴选中（在当前鼠标位置）。"""
    _notify_timer_shield(400)
    pydirectinput.keyDown(_KEY_PAUSE)
    time.sleep(0.01)
    pydirectinput.click(button='left')
    pydirectinput.keyDown('esc')

    time.sleep(0.1)
    pydirectinput.keyUp('esc')
    pydirectinput.keyUp(_KEY_PAUSE)


def _bind_hotkeys():
    global _keyboard_listener
    stop_matchstick_listener()

    def _on_press(key):
        try:
            key_name = None
            if hasattr(key, 'char') and key.char is not None:
                key_name = key.char.lower()
            elif hasattr(key, 'name') and key.name is not None:
                key_name = key.name.lower()

            if _MATCHSTICK_ENABLED.get("pass_166ms") and key_name == _MATCHSTICK_HOTKEYS.get("pass_166ms", "").lower():
                p_and_esc_click()
            elif _MATCHSTICK_ENABLED.get("select_operator") and key_name == _MATCHSTICK_HOTKEYS.get("select_operator", "").lower():
                p_and_left_click()
            elif _MATCHSTICK_ENABLED.get("pass_50ms") and key_name == _MATCHSTICK_HOTKEYS.get("pass_50ms", "").lower():
                p_and_esc_click_short()
        except Exception:
            pass

    try:
        _keyboard_listener = Listener(on_press=_on_press)
        _keyboard_listener.start()
    except Exception:
        pass


def start_matchstick_listener():
    """启动划火柴全局热键监听（非阻塞）。"""
    _bind_hotkeys()


def stop_matchstick_listener():
    """停止划火柴全局热键监听。"""
    global _keyboard_listener
    if _keyboard_listener is not None:
        try:
            _keyboard_listener.stop()
            _keyboard_listener.join(timeout=1.0)
        except Exception:
            pass
        _keyboard_listener = None


def run_hotkey_listener():
    _bind_hotkeys()
    if _keyboard_listener is not None:
        _keyboard_listener.join()


if __name__ == "__main__":
    run_hotkey_listener()
