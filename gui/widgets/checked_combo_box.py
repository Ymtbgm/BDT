from PyQt6.QtCore import Qt, QEvent, pyqtSignal
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import QComboBox, QAbstractItemView


class CheckedComboBox(QComboBox):
    """支持多选勾选的伪下拉框，显示已勾选项的文本。"""

    item_changed = pyqtSignal()

    def __init__(self, placeholder: str = "请选择", parent=None):
        super().__init__(parent)
        self._placeholder = placeholder

        model = QStandardItemModel(self)
        self.setModel(model)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setPlaceholderText(placeholder)
        self.lineEdit().setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        view = self.view()
        view.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        view.viewport().installEventFilter(self)

        model.itemChanged.connect(self._update_text)
        self._update_text()

    def add_item(self, label: str, data: str, checked: bool = False):
        item = QStandardItem(label)
        item.setCheckable(True)
        item.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        item.setData(data, Qt.ItemDataRole.UserRole)
        # 只保留勾选交互，不要选中高亮
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.model().appendRow(item)
        self._update_text()

    def checked_data(self) -> list[str]:
        result = []
        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            if item.checkState() == Qt.CheckState.Checked:
                value = item.data(Qt.ItemDataRole.UserRole)
                if value is not None:
                    result.append(value)
        return result

    def set_checked_data(self, keys: list[str]):
        key_set = set(keys)
        model = self.model()
        for i in range(model.rowCount()):
            item = model.item(i)
            value = item.data(Qt.ItemDataRole.UserRole)
            state = Qt.CheckState.Checked if value in key_set else Qt.CheckState.Unchecked
            item.setCheckState(state)
        self._update_text()

    def is_checked(self, key: str) -> bool:
        return key in self.checked_data()

    def _update_text(self):
        checked = [
            self.model().item(i).text()
            for i in range(self.model().rowCount())
            if self.model().item(i).checkState() == Qt.CheckState.Checked
        ]
        text = ", ".join(checked) if checked else self._placeholder
        self.lineEdit().setText(text)
        self.item_changed.emit()

    def eventFilter(self, obj, event):
        if obj is self.view().viewport() and event.type() == QEvent.Type.MouseButtonRelease:
            index = self.view().indexAt(event.pos())
            if index.isValid():
                item = self.model().itemFromIndex(index)
                if item.isCheckable():
                    new_state = (
                        Qt.CheckState.Unchecked
                        if item.checkState() == Qt.CheckState.Checked
                        else Qt.CheckState.Checked
                    )
                    item.setCheckState(new_state)
                    return True
            # 点击空白处也保持下拉框打开
            return True
        return super().eventFilter(obj, event)
