import os
import sys

# 避免 Qt 重复设置 DPI awareness 导致 Windows "拒绝访问" 警告。
# 若进程已被其他库设置为某种 DPI awareness，Qt 的默认设置会失败；
# 这里通过关闭 qt.qpa.window 日志来抑制该提示。
os.environ.setdefault("QT_QPA_PLATFORM", "windows:dpiawareness=0")
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.window=false")

from PyQt6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
