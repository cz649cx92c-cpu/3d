#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import rospy
from std_msgs.msg import String
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PcdImportWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.publisher = rospy.Publisher("/pcd_file_cmd", String, queue_size=1, latch=True)
        self.setWindowTitle("PCD Map Import")
        self.resize(620, 220)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        row.addWidget(QLabel("PCD file"))
        self.path_edit = QLineEdit(str(Path.home()))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row.addWidget(self.path_edit, 1)
        row.addWidget(browse)
        layout.addLayout(row)

        button_row = QHBoxLayout()
        import_btn = QPushButton("Import PCD to OctoMap pipeline")
        import_btn.clicked.connect(self._import_pcd)
        button_row.addWidget(import_btn)
        layout.addLayout(button_row)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Import log")
        layout.addWidget(self.log_view, 1)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select PCD file", str(Path.home()), "PCD (*.pcd)")
        if path:
            self.path_edit.setText(path)

    def _import_pcd(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.information(self, "Info", "Choose a PCD file first.")
            return
        if not Path(path).exists():
            QMessageBox.warning(self, "Missing file", f"PCD file not found:\n{path}")
            return
        self.publisher.publish(String(data=path))
        self.log_view.append(f"Published /pcd_file_cmd: {path}")


def main() -> None:
    rospy.init_node("pcd_map_import_gui", disable_signals=True)
    app = QApplication(sys.argv)
    window = PcdImportWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
