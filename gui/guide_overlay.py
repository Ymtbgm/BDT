from __future__ import annotations

import re

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QPoint, QSize, QRect, QTimer
from PyQt6.QtGui import QPixmap, QFont

from core.base.paths import gui_template
from gui._window_effects import remove_dwm_glass_border, set_window_topmost


class HighlightOverlay(QWidget):
    """用于在引导过程中高亮目标控件/标签的红色边框浮层。"""

    def __init__(self, parent=None):
        super().__init__(
            parent,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        from PyQt6.QtGui import QPainter, QPen, QColor
        painter = QPainter(self)
        pen = QPen(QColor(255, 0, 0))
        pen.setWidth(3)
        painter.setPen(pen)
        # 内缩 1px，避免边框被窗口边缘裁切
        painter.drawRect(self.rect().adjusted(1, 1, -1, -1))
        painter.end()

    def showEvent(self, event):
        super().showEvent(event)
        remove_dwm_glass_border(self)
        set_window_topmost(self)


class GuideOverlay(QWidget):
    """首次启动引导浮层，显示在 MainWindow 右侧。

    浮层包含 guide.png 角色形象和 dialog_box.png 对话框，
    按顺序依次介绍各个 tab，用户可跳过或继续。
    """

    # 引导版本号，更新引导流程后递增，用户会重新看到新引导
    GUIDE_VERSION = 1

    # 引导步骤：(tab_index, title, content)
    STEPS: list[tuple[int, str, str]] = [
        (
            0,
            "引导",
            "欢迎使用BDT，接下来将带你了解各项功能~"
        ),
        (
            0,
            "脚本执行",
            "这里是<span style='color:red'>脚本执行</span>页面。<br>"
            "当你有一个完整的轴的脚本后，"
            "可以通过<span style='color:red'>脚本选择</span>选中来进行自动凸图。"
        ),
        (
            0,
            "脚本执行",
            "同时左侧的<span style='color:red'>键位配置</span>"
            "记得配置为和游戏一致喔。"
        ),
        (
            1,
            "脚本编辑",
            "这里是<span style='color:red'>脚本编辑</span>页面。<br>"
            "你可以新建或打开现有的脚本，进行更改。"
        ),
        (
            1,
            "脚本编辑",
            "在大多数时间你或许不需要从头创建一个脚本，在后续<span style='color:red'>操作录制</span>页面，"
            "你能够在游戏的同时就获得一个复制你操作的脚本。"
        ),
        (
            2,
            "资源更新",
            "这里是<span style='color:red'>资源更新</span>页面。"
            "你可以手动替换，也可以开启自动检查与下载。",
        ),
        (
            3,
            "划火柴",
            "这里是<span style='color:red'>划火柴</span>页面。提供了一些<span style='color:red'>键位设置</span>，"
            "不过请在子弹时间进行操作，按照0.2倍率计算你就会明白。",
        ),
        (
            4,
            "计时器",
            "这里是<span style='color:red'>计时器</span>页面。选择费用条 tag 后启动，工具会通过识别费用条"
            "或倍率区来精确同步游戏内时间。",
        ),
        (
            4,
            "计时器",
            "即使在费用条满后也会通过倍率区继续计时，请注意使用<span style='color:red'>快捷键暂停</span>，"
            "为了尽可能的精度，并没有设置点击暂停的判定。",
        ),
        (
            5,
            "操作录制",
            "这里是<span style='color:red'>操作录制</span>页面。他应该会是你最常使用的界面。"
        ),
        (
            5,
            "操作录制",
            "配置好<span style='color:red'>初始干员数量</span>和"
            "<span style='color:red'>初始道具数量</span>就可以直接在编队界面点击<span style='color:red'>开始录制</span>。"
        ),
        (
            5,
            "操作录制",
            "首次启动OCR可能较慢，"
            "当悬浮窗提示<span style='color:red'>识别完成，可以进入作战</span>后，便可以直接进入作战。点击结束录制后，你将会获得一份刚才操作的脚本"
        ),
        (
            5,
            "操作录制",
            "脚本录制有时也会出现一些偏差，你可以在<span style='color:red'>脚本编辑</span>页面中修正。"
        ),
        (
            5,
            "操作录制",
            "哪怕你还没有一份完整的轴，你也可以直接启用<span style='color:red'>脚本装载</span>。"
            "装载后在编队界面点击开始录制，脚本自动执行到结束并暂停。"
        ),
        (
            5,
            "操作录制",
            "你可以在脚本执行的任何时刻进行<span style='color:red'>接管</span>，或在脚本结束后直接继续操作，录制器会继续录制你的所有操作。"
        ),
        (
            5,
            "操作录制",
            "这意味着你可以在任何时候记录所有轴，减少重复性工作。"
            "<span style='color:red'>即使你只摸完了前10%的轴，也不需要重复前10%的轴的重复操作</span>。"
        ),
        (
            5,
            "操作录制",
            "如果有需要凸概率的地方，你也可以在<span style='color:red'>脚本编辑</span>中"
            "为脚本添加概率点，脚本会自动执行到概率满足为止。"
        ),
        (
            5,
            "感谢",
            "目前这仍然是一个测试中版本，如果你有<span style='color:red'>任何使用疑惑或者建议和bug</span>可以积极反馈，十分感谢~"
        ),
    ]

    # 红色文本 -> 高亮目标映射
    # 格式：("tab", index) / ("widget", "name") / ("widgets", ["name1", "name2"])
    HIGHLIGHT_TARGETS: dict[str, tuple[str, int | str | list[str]]] = {
        # tab 标签
        "脚本执行": ("tab", 0),
        "脚本编辑": ("tab", 1),
        "资源更新": ("tab", 2),
        "划火柴": ("tab", 3),
        "计时器": ("tab", 4),
        "操作录制": ("tab", 5),
        # box/控件
        "脚本选择": ("widget", "script_group"),
        "键位配置": ("widget", "exec_keys_group"),
        "键位设置": ("widget", "matchstick_keys_group"),
        "开始录制": ("widget", "rec_exec_group"),
        # 控件 + 标题合并高亮
        "初始干员数量": ("widgets", ("rec_initial_operator_count_label", "rec_initial_operator_count")),
        "初始道具数量": ("widgets", ("rec_initial_item_count_label", "rec_initial_item_count")),
        "脚本装载": ("widgets", ("rec_loaded_script_label", "rec_loaded_script_path", "btn_rec_load_script")),
        "接管": ("widgets", ("rec_takeover_hotkey_label", "rec_takeover_hotkey")),
    }

    def __init__(self, main_window: QWidget):
        super().__init__(
            main_window,
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Window
            | Qt.WindowType.WindowDoesNotAcceptFocus,
        )
        self.main_window = main_window
        self._step_index = 0

        self._dialog_height = 220
        self._guide_height = 300
        self._fixed_width = 600
        # 整体内容向下偏移量（guide 和 dialog_box 一起下移）
        self._content_top_offset = 50

        self._build_ui()
        self._update_step()
        self._position_to_main_window()

    def _build_ui(self):
        self.setFixedSize(self._fixed_width, self._dialog_height + self._guide_height + 20)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)

        # 高亮框浮层列表（支持多个控件分别高亮）
        self._highlight_overlays: list[HighlightOverlay] = []

        # --- 角色形象区域（先创建，使对话框叠在角色上方） ---
        self.guide_label = QLabel(self)
        self.guide_label.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        guide_pixmap = QPixmap(str(gui_template("guide.png")))
        if not guide_pixmap.isNull():
            scaled = guide_pixmap.scaled(
                QSize(self._fixed_width - 40, self._guide_height - 20),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.guide_label.setPixmap(scaled)
            self.guide_label.setFixedSize(scaled.size())
        # 朝右 250px，朝上 30px（与对话框底部重叠），并叠加整体下移偏移
        self.guide_label.move(
            250, self._dialog_height - 30 + self._content_top_offset
        )

        # --- 对话框区域 ---
        self.dialog_container = QWidget(self)
        # 宽度先设为固定值，后续根据实际缩放后的 dialog_box 图片调整
        self.dialog_container.setFixedSize(self._fixed_width, self._dialog_height)
        self.dialog_container.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.dialog_container.move(0, 0)
        dialog_layout = QVBoxLayout(self.dialog_container)
        # 右边距加 50 使按钮左移，底加 30 使按钮上移
        dialog_layout.setContentsMargins(25, 35, 50, 50)
        dialog_layout.setSpacing(8)

        self.title_label = QLabel()
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        self.title_label.setStyleSheet("color: #5a1a1a;")
        dialog_layout.addWidget(self.title_label)

        self.content_label = QLabel()
        self.content_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.content_label.setWordWrap(True)
        content_font = QFont()
        content_font.setPointSize(10)
        content_font.setBold(True)
        self.content_label.setFont(content_font)
        self.content_label.setStyleSheet("color: #333333;")
        self.content_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        dialog_layout.addWidget(self.content_label)

        # 按钮行
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        self.skip_btn = QPushButton("跳过引导")
        self.skip_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.skip_btn.clicked.connect(self._skip_guide)
        self.continue_btn = QPushButton("开始引导")
        self.continue_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.continue_btn.clicked.connect(self._next_step)
        self.continue_btn.setDefault(True)
        btn_layout.addStretch()
        btn_layout.addWidget(self.skip_btn)
        btn_layout.addWidget(self.continue_btn)
        dialog_layout.addLayout(btn_layout)

        # --- 对话框背景 ---
        self._dialog_pixmap = QPixmap(str(gui_template("dialog_box.png")))
        if not self._dialog_pixmap.isNull():
            # 先裁剪掉上下/左右透明边，再按比例缩放到容器
            self._dialog_pixmap = self._crop_to_content(self._dialog_pixmap)
            self._dialog_pixmap = self._dialog_pixmap.scaled(
                QSize(self._fixed_width, self._dialog_height),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            # 让对话框容器尺寸与图片完全一致，避免四周出现多余颜色
            self.dialog_container.setFixedSize(
                self._dialog_pixmap.width(), self._dialog_pixmap.height()
            )
            self.dialog_container.move(
                (self._fixed_width - self._dialog_pixmap.width()) // 2,
                self._content_top_offset,
            )

        self.dialog_container.paintEvent = self._paint_dialog_background

    def _crop_to_content(self, pixmap: QPixmap) -> QPixmap:
        """裁剪掉图片四周的透明像素，返回紧凑内容。"""
        image = pixmap.toImage()
        w, h = image.width(), image.height()
        left, top, right, bottom = w, h, 0, 0
        threshold = 30
        for y in range(h):
            for x in range(w):
                if image.pixelColor(x, y).alpha() > threshold:
                    if x < left:
                        left = x
                    if x > right:
                        right = x
                    if y < top:
                        top = y
                    if y > bottom:
                        bottom = y
        if right < left or bottom < top:
            return pixmap
        # 留出 2px 边距，避免抗锯齿边缘被切
        left = max(0, left - 2)
        top = max(0, top - 2)
        right = min(w - 1, right + 2)
        bottom = min(h - 1, bottom + 2)
        return pixmap.copy(left, top, right - left + 1, bottom - top + 1)

    def paintEvent(self, event):
        """清除窗口背景，防止透明窗口上子控件移动后产生残影。"""
        from PyQt6.QtGui import QPainter
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
        painter.eraseRect(self.rect())
        painter.end()
        super().paintEvent(event)

    def showEvent(self, event):
        """窗口显示后禁用 Win11 DWM 玻璃材质并置顶。"""
        super().showEvent(event)
        remove_dwm_glass_border(self)
        set_window_topmost(self)

    def _paint_dialog_background(self, event):
        from PyQt6.QtGui import QPainter
        painter = QPainter(self.dialog_container)
        if self._dialog_pixmap and not self._dialog_pixmap.isNull():
            # 容器尺寸已与图片一致，直接绘制
            painter.drawPixmap(0, 0, self._dialog_pixmap)
        painter.end()

    def _update_step(self):
        if self._step_index >= len(self.STEPS):
            self._finish_guide()
            return

        tab_index, title, content = self.STEPS[self._step_index]
        self.title_label.setText(f"{title}")
        self.content_label.setText(content)

        if self._step_index == 0:
            self.continue_btn.setText("开始引导")
            self.skip_btn.setVisible(True)
        elif self._step_index == len(self.STEPS) - 1:
            self.continue_btn.setText("完成")
            self.skip_btn.setVisible(False)
        else:
            self.continue_btn.setText("继续")
            self.skip_btn.setVisible(True)

        # 切换到对应 tab
        tabs = getattr(self.main_window, "_tabs", None)
        if tabs is not None and tab_index < tabs.count():
            tabs.setCurrentIndex(tab_index)

        # 延迟一帧更新高亮框，等 tab 切换和控件布局完成后再计算位置
        QTimer.singleShot(0, lambda: self._update_highlight(title, content))

    def _update_highlight(self, title: str, content: str):
        """根据当前步骤内容中的红色文本，高亮对应控件/tab。"""
        targets = self._find_highlight_targets(title, content)

        # 隐藏多余的高亮框
        for i in range(len(targets), len(self._highlight_overlays)):
            self._highlight_overlays[i].hide()

        if not targets:
            return

        # 为每个目标创建/复用一个高亮框
        for idx, target in enumerate(targets):
            rect = self._get_target_rect(target)
            if rect is None or rect.isEmpty():
                if idx < len(self._highlight_overlays):
                    self._highlight_overlays[idx].hide()
                continue

            if idx >= len(self._highlight_overlays):
                overlay = HighlightOverlay(self.main_window)
                self._highlight_overlays.append(overlay)
            else:
                overlay = self._highlight_overlays[idx]

            overlay.setGeometry(rect)
            overlay.show()
            # 确保窗口显示后再应用 DWM/置顶
            remove_dwm_glass_border(overlay)
            set_window_topmost(overlay)

        # 高亮框创建/置顶后，重新把引导浮层置顶，确保 guide 和 dialog 在最上层
        set_window_topmost(self)

    def _find_highlight_targets(
        self, title: str, content: str
    ) -> list[tuple[str, int | str]]:
        """从 title 和 content 中提取红色文本，查找所有对应高亮目标。"""
        targets = []
        seen = set()
        # 匹配 <span style='color:red'>...</span> 或 <font color='red'>...</font>
        patterns = [
            r"<span[^>]*color\s*[=:]\s*['\"]?red['\"]?[^>]*>(.*?)</span>",
            r"<font[^>]*color\s*=\s*['\"]?red['\"]?[^>]*>(.*?)</font>",
        ]
        for text in (title, content):
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    keyword = re.sub(r"<[^>]+>", "", match.group(1)).strip()
                    if keyword in self.HIGHLIGHT_TARGETS:
                        target = self.HIGHLIGHT_TARGETS[keyword]
                        if target not in seen:
                            seen.add(target)
                            targets.append(target)
        return targets

    def _get_target_rect(self, target: tuple[str, int | str | list[str]]) -> QRect | None:
        """获取目标在屏幕上的几何区域。"""
        if self.main_window is None:
            return None

        kind, value = target
        if kind == "tab":
            tabs = getattr(self.main_window, "_tabs", None)
            if tabs is None:
                return None
            tab_bar = tabs.tabBar()
            if value >= tab_bar.count():
                return None
            local_rect = tab_bar.tabRect(value)
            top_left = tab_bar.mapToGlobal(local_rect.topLeft())
            return QRect(top_left, local_rect.size())

        elif kind == "widget":
            widget = getattr(self.main_window, str(value), None)
            if widget is None:
                return None
            top_left = widget.mapToGlobal(QPoint(0, 0))
            return QRect(top_left, widget.size())

        elif kind == "widgets":
            united_rect: QRect | None = None
            for name in value:
                widget = getattr(self.main_window, str(name), None)
                if widget is None:
                    continue
                top_left = widget.mapToGlobal(QPoint(0, 0))
                rect = QRect(top_left, widget.size())
                united_rect = rect if united_rect is None else united_rect.united(rect)
            return united_rect

        return None

    def _next_step(self):
        self._step_index += 1
        self._update_step()

    def _skip_guide(self):
        self._mark_shown()
        for overlay in self._highlight_overlays:
            overlay.close()
        self.close()

    def _finish_guide(self):
        self._mark_shown()
        for overlay in self._highlight_overlays:
            overlay.close()
        self.close()

    def _mark_shown(self):
        """通知主窗口记录当前引导版本号。"""
        if self.main_window is not None:
            self.main_window._mark_guide_shown(self.GUIDE_VERSION)

    def _position_to_main_window(self):
        if not self.main_window:
            return
        geo = self.main_window.geometry()
        # 向左偏移 250px，让对话框更贴近主窗口
        x = geo.x() + geo.width() - 350
        y = geo.y()
        self.move(QPoint(x, y))

    def moveEvent(self, event):
        """浮层移动时保持与主窗口的相对位置（可选）。"""
        super().moveEvent(event)
