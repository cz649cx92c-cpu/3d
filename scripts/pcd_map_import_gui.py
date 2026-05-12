#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import rospy
from std_msgs.msg import String
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PcdImportWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.publisher = rospy.Publisher("/pcd_file_cmd", String, queue_size=1, latch=True)
        self.setWindowTitle("PCD Map Import")
        self.resize(720, 520)
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

        params_group = QGroupBox("Import Parameters")
        params_form = QFormLayout(params_group)

        self.resolution_spin = QDoubleSpinBox()
        self.resolution_spin.setRange(0.05, 5.0)
        self.resolution_spin.setSingleStep(0.05)
        self.resolution_spin.setValue(0.5)
        params_form.addRow("Voxel size (m)", self.resolution_spin)

        self.downsample_spin = QDoubleSpinBox()
        self.downsample_spin.setRange(0.0, 5.0)
        self.downsample_spin.setSingleStep(0.05)
        self.downsample_spin.setValue(0.3)
        params_form.addRow("Downsample (m)", self.downsample_spin)

        self.min_points_spin = QSpinBox()
        self.min_points_spin.setRange(1, 100)
        self.min_points_spin.setValue(1)
        params_form.addRow("Min points / voxel", self.min_points_spin)

        self.min_cluster_spin = QSpinBox()
        self.min_cluster_spin.setRange(1, 500)
        self.min_cluster_spin.setValue(1)
        params_form.addRow("Min cluster voxels", self.min_cluster_spin)

        self.outer_shell_spin = QSpinBox()
        self.outer_shell_spin.setRange(0, 20)
        self.outer_shell_spin.setValue(1)
        params_form.addRow("Remove outer wall layers", self.outer_shell_spin)

        self.max_voxels_spin = QSpinBox()
        self.max_voxels_spin.setRange(10000, 5000000)
        self.max_voxels_spin.setSingleStep(50000)
        self.max_voxels_spin.setValue(300000)
        params_form.addRow("Safety voxel limit", self.max_voxels_spin)

        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(-10.0, 10.0)
        self.z_min_spin.setSingleStep(0.1)
        self.z_min_spin.setValue(-0.2)
        params_form.addRow("Keep z >= (m)", self.z_min_spin)

        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(-10.0, 10.0)
        self.z_max_spin.setSingleStep(0.1)
        self.z_max_spin.setValue(1.2)
        params_form.addRow("Keep z <= (m)", self.z_max_spin)

        layout.addWidget(params_group)

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
        payload = {
            "path": path,
            "resolution": self.resolution_spin.value(),
            "voxel_downsample_m": self.downsample_spin.value(),
            "min_points_per_voxel": self.min_points_spin.value(),
            "min_cluster_voxels": self.min_cluster_spin.value(),
            "outer_shell_layers": self.outer_shell_spin.value(),
            "max_occupied_voxels": self.max_voxels_spin.value(),
            "z_min": self.z_min_spin.value(),
            "z_max": self.z_max_spin.value(),
        }
        self.publisher.publish(String(data=json.dumps(payload, ensure_ascii=False)))
        self.log_view.append(
            "Published /pcd_file_cmd:\n"
            f"  path={path}\n"
            f"  voxel={payload['resolution']:.2f} downsample={payload['voxel_downsample_m']:.2f} "
            f"min_pts={payload['min_points_per_voxel']} min_cluster={payload['min_cluster_voxels']}\n"
            f"  outer_shell={payload['outer_shell_layers']} z_range=[{payload['z_min']:.2f}, {payload['z_max']:.2f}] "
            f"limit={payload['max_occupied_voxels']}"
        )


def main() -> None:
    rospy.init_node("pcd_map_import_gui", disable_signals=True)
    app = QApplication(sys.argv)
    window = PcdImportWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
