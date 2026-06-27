from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QCheckBox, QTextEdit,
)


class MatchstickTab(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # 操作行：选中干员
        row_select = QHBoxLayout()
        row_select.addWidget(QLabel("选中干员"))
        self.main_window.chk_matchstick_select = QCheckBox("启用")
        self.main_window.chk_matchstick_select.stateChanged.connect(self.main_window._on_matchstick_enabled_changed)
        row_select.addWidget(self.main_window.chk_matchstick_select)
        self.main_window.line_matchstick_select = QLineEdit("r")
        self.main_window.line_matchstick_select.setMaxLength(16)
        self.main_window.line_matchstick_select.setFixedWidth(100)
        self.main_window.line_matchstick_select.textChanged.connect(self.main_window._on_matchstick_hotkey_changed)
        row_select.addWidget(self.main_window.line_matchstick_select)
        row_select.addStretch()
        layout.addLayout(row_select)

        # 操作行：过 166ms
        row_166 = QHBoxLayout()
        row_166.addWidget(QLabel("过 166ms"))
        self.main_window.chk_matchstick_166 = QCheckBox("启用")
        self.main_window.chk_matchstick_166.stateChanged.connect(self.main_window._on_matchstick_enabled_changed)
        row_166.addWidget(self.main_window.chk_matchstick_166)
        self.main_window.line_matchstick_166 = QLineEdit("space")
        self.main_window.line_matchstick_166.setMaxLength(16)
        self.main_window.line_matchstick_166.setFixedWidth(100)
        self.main_window.line_matchstick_166.textChanged.connect(self.main_window._on_matchstick_hotkey_changed)
        row_166.addWidget(self.main_window.line_matchstick_166)
        row_166.addStretch()
        layout.addLayout(row_166)

        # 操作行：过 50ms
        row_50 = QHBoxLayout()
        row_50.addWidget(QLabel("过 50ms"))
        self.main_window.chk_matchstick_50 = QCheckBox("启用")
        self.main_window.chk_matchstick_50.stateChanged.connect(self.main_window._on_matchstick_enabled_changed)
        row_50.addWidget(self.main_window.chk_matchstick_50)
        self.main_window.line_matchstick_50 = QLineEdit("f")
        self.main_window.line_matchstick_50.setMaxLength(16)
        self.main_window.line_matchstick_50.setFixedWidth(100)
        self.main_window.line_matchstick_50.textChanged.connect(self.main_window._on_matchstick_hotkey_changed)
        row_50.addWidget(self.main_window.line_matchstick_50)
        row_50.addStretch()
        layout.addLayout(row_50)

        # 说明文本
        info = QTextEdit()
        info.setReadOnly(True)
        info.setHtml(
            """
            <h3>划火柴快捷键</h3>
            <p>在此绑定全局热键，开启后可在游戏中快速执行划火柴操作，为了减小电脑端误差，<span style="color: red;">所有操作请在子弹时间下进行</span>，否则不是刚好过1帧和0帧选取。</p>
            <ul>
                <li><b>选中干员</b>：用于选中鼠标指向的干员/装置。</li>
                <li><b>过 166ms</b>： 推进166ms，子弹时间下即为33ms，用于在子弹时间中推进一帧。</li>
                <li><b>过 50ms</b>：推进50ms，子弹时间下即为10ms，补齐不到1帧的时间。</li>
            </ul>
            <p><span style="color: red;">热键不可与脚本执行页中的暂停/技能/撤退键相同，否则会造成循环触发。</span></p>
            """
        )
        layout.addWidget(info)
        layout.addStretch()
