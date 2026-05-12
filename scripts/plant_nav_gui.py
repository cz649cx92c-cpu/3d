#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from PIL.ImageQt import toqimage
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen, QPixmap, QPolygonF
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from plant_nav_gui.map_model import (  # noqa: E402
    MapState,
    NoGoZone,
    Pose2D,
    RouteRecord,
    load_ascii_pcd_projection,
    load_map_package,
    load_ros_map,
    load_world_projection,
    plan_path,
    save_map_package,
)
from plant_nav_gui.ros_bridge import ROSBridge  # noqa: E402


class ImportWorker(QThread):
    finished_ok = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, loader, parent=None):
        super().__init__(parent)
        self.loader = loader

    def run(self) -> None:
        try:
            result = self.loader()
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


class MapCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.state: MapState | None = None
        self.mode = "idle"
        self.preview_zone: list[Pose2D] = []
        self.robot_pose: Pose2D | None = None
        self.setMinimumSize(640, 640)
        self.setMouseTracking(True)

    def set_state(self, state: MapState | None) -> None:
        self.state = state
        self.preview_zone = []
        self.update()

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        if mode != "zone":
            self.preview_zone = []
        self.update()

    def set_robot_pose(self, point: Pose2D | None) -> None:
        self.robot_pose = point
        self.update()

    def _map_rect(self) -> QRectF | None:
        if self.state is None:
            return None
        occupancy = self.state.occupancy
        margin = 20.0
        available_w = max(10.0, self.width() - margin * 2.0)
        available_h = max(10.0, self.height() - margin * 2.0)
        scale = min(available_w / occupancy.width, available_h / occupancy.height)
        draw_w = occupancy.width * scale
        draw_h = occupancy.height * scale
        left = (self.width() - draw_w) / 2.0
        top = (self.height() - draw_h) / 2.0
        return QRectF(left, top, draw_w, draw_h)

    def _world_to_widget(self, point: Pose2D) -> QPointF | None:
        if self.state is None:
            return None
        rect = self._map_rect()
        if rect is None:
            return None
        occupancy = self.state.occupancy
        gx = (point.x - occupancy.origin_x) / occupancy.resolution
        gy = (point.y - occupancy.origin_y) / occupancy.resolution
        px = rect.left() + gx * (rect.width() / occupancy.width)
        py = rect.bottom() - gy * (rect.height() / occupancy.height)
        return QPointF(px, py)

    def _widget_to_world(self, event_pos) -> Pose2D | None:
        if self.state is None:
            return None
        rect = self._map_rect()
        if rect is None or not rect.contains(QPointF(event_pos)):
            return None
        occupancy = self.state.occupancy
        gx = (event_pos.x() - rect.left()) / (rect.width() / occupancy.width)
        gy = (rect.bottom() - event_pos.y()) / (rect.height() / occupancy.height)
        return Pose2D(
            occupancy.origin_x + gx * occupancy.resolution,
            occupancy.origin_y + gy * occupancy.resolution,
        )

    def mousePressEvent(self, event):  # noqa: N802
        if self.state is None or event.button() != Qt.LeftButton:
            return
        point = self._widget_to_world(event.pos())
        if point is None:
            return
        if self.mode == "start":
            self.state.start = point
            self.update()
            window = self.window()
            if hasattr(window, "on_canvas_point_set"):
                window.on_canvas_point_set("start")
        elif self.mode == "goal":
            self.state.goal = point
            self.update()
            window = self.window()
            if hasattr(window, "on_canvas_point_set"):
                window.on_canvas_point_set("goal")
        elif self.mode == "zone":
            self.preview_zone.append(point)
            self.update()

    def paintEvent(self, _event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#11161b"))
        if self.state is None:
            painter.setPen(QColor("#d6dee7"))
            painter.drawText(self.rect(), Qt.AlignCenter, "Load a map to view the local planning canvas")
            return
        rect = self._map_rect()
        occupancy = self.state.occupancy
        image = occupancy.to_pil_image()
        qimage = toqimage(image)
        pixmap = QPixmap.fromImage(qimage).scaled(int(rect.width()), int(rect.height()), Qt.IgnoreAspectRatio, Qt.FastTransformation)
        painter.drawPixmap(rect.toRect(), pixmap)

        grid_pen = QPen(QColor(255, 255, 255, 30))
        painter.setPen(grid_pen)
        painter.drawRect(rect)

        zone_pen = QPen(QColor("#ff8c42"), 2)
        zone_fill = QColor(255, 140, 66, 70)
        for zone in self.state.zones:
            polygon = QPolygonF()
            for point in zone.points:
                widget_pt = self._world_to_widget(point)
                if widget_pt is not None:
                    polygon.append(widget_pt)
            if polygon.count() >= 3:
                painter.setPen(zone_pen)
                painter.setBrush(zone_fill)
                painter.drawPolygon(polygon)

        if self.preview_zone:
            polygon = QPolygonF()
            for point in self.preview_zone:
                widget_pt = self._world_to_widget(point)
                if widget_pt is not None:
                    polygon.append(widget_pt)
            painter.setPen(QPen(QColor("#ffd166"), 2, Qt.DashLine))
            painter.setBrush(QColor(255, 209, 102, 40))
            if polygon.count() >= 2:
                painter.drawPolyline(polygon)

        self._draw_points(painter, self.state.planned_path, QColor("#43d17a"), 3)
        for route in self.state.routes:
            self._draw_points(painter, route.points, QColor("#5ac8fa"), 2)
        if self.state.start:
            self._draw_marker(painter, self.state.start, QColor("#4caf50"), "S")
        if self.state.goal:
            self._draw_marker(painter, self.state.goal, QColor("#f44336"), "G")
        if self.robot_pose:
            self._draw_marker(painter, self.robot_pose, QColor("#00e5ff"), "R")

    def _draw_points(self, painter: QPainter, points: list[Pose2D], color: QColor, width: int) -> None:
        if len(points) < 2:
            return
        poly = QPolygonF()
        for point in points:
            widget_pt = self._world_to_widget(point)
            if widget_pt is not None:
                poly.append(widget_pt)
        if poly.count() >= 2:
            painter.setPen(QPen(color, width))
            painter.setBrush(Qt.NoBrush)
            painter.drawPolyline(poly)

    def _draw_marker(self, painter: QPainter, point: Pose2D, color: QColor, text: str) -> None:
        widget_pt = self._world_to_widget(point)
        if widget_pt is None:
            return
        painter.setPen(QPen(color, 2))
        painter.setBrush(color)
        painter.drawEllipse(widget_pt, 7, 7)
        painter.setPen(QColor("white"))
        painter.drawText(QRectF(widget_pt.x() - 10, widget_pt.y() - 24, 20, 14), Qt.AlignCenter, text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.state: MapState | None = None
        self.bridge = ROSBridge()
        self.recording_active = False
        self.record_buffer: list[Pose2D] = []
        self.auto_drive_process: subprocess.Popen | None = None
        self.import_worker: ImportWorker | None = None
        self.setWindowTitle("Plant Navigation GUI")
        self.resize(1380, 880)
        self._build_ui()
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_robot_pose)
        self._poll_timer.start(250)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(12, 12, 12, 12)

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)

        info_group = QGroupBox("Status")
        info_form = QFormLayout(info_group)
        self.map_name_label = QLabel("Not loaded")
        self.source_label = QLabel("Not loaded")
        self.ros_status_label = QLabel(self.bridge.status().message)
        info_form.addRow("Map", self.map_name_label)
        info_form.addRow("Source", self.source_label)
        info_form.addRow("ROS", self.ros_status_label)
        left_layout.addWidget(info_group)

        import_group = QGroupBox("Map Import / Package")
        import_layout = QVBoxLayout(import_group)
        ros_map_button = QPushButton("Import ROS 2D map (.yaml)")
        ros_map_button.clicked.connect(self.import_ros_map)
        pcd_button = QPushButton("Import PCD occupancy map")
        pcd_button.clicked.connect(self.import_pcd)
        world_button = QPushButton("Import Gazebo .world / .sdf")
        world_button.clicked.connect(self.import_world)
        load_pkg_button = QPushButton("Load map package")
        load_pkg_button.clicked.connect(self.load_package)
        save_pkg_button = QPushButton("Save map package")
        save_pkg_button.clicked.connect(self.save_package)
        import_layout.addWidget(ros_map_button)
        import_layout.addWidget(pcd_button)
        import_layout.addWidget(world_button)
        import_layout.addWidget(load_pkg_button)
        import_layout.addWidget(save_pkg_button)
        left_layout.addWidget(import_group)

        plan_group = QGroupBox("Planning / No-Go Zones")
        plan_layout = QVBoxLayout(plan_group)
        set_start_button = QPushButton("Pick start point")
        set_start_button.clicked.connect(lambda: self.set_canvas_mode("start"))
        set_goal_button = QPushButton("Pick goal point")
        set_goal_button.clicked.connect(lambda: self.set_canvas_mode("goal"))
        zone_start_button = QPushButton("Draw plant no-go zone")
        zone_start_button.clicked.connect(lambda: self.set_canvas_mode("zone"))
        zone_finish_button = QPushButton("Finish current zone")
        zone_finish_button.clicked.connect(self.finish_zone)
        zone_clear_button = QPushButton("Clear all zones")
        zone_clear_button.clicked.connect(self.clear_zones)
        plan_button = QPushButton("Plan path")
        plan_button.clicked.connect(self.plan_current_path)
        clear_path_button = QPushButton("Clear planned path")
        clear_path_button.clicked.connect(self.clear_planned_path)
        for button in [set_start_button, set_goal_button, zone_start_button, zone_finish_button, zone_clear_button, plan_button, clear_path_button]:
            plan_layout.addWidget(button)
        left_layout.addWidget(plan_group)

        route_group = QGroupBox("Route Recording / Paths")
        route_layout = QVBoxLayout(route_group)
        self.route_list = QListWidget()
        start_record_button = QPushButton("Start /odom recording")
        start_record_button.clicked.connect(self.start_recording)
        stop_record_button = QPushButton("Stop and save recording")
        stop_record_button.clicked.connect(self.stop_recording)
        save_planned_route_button = QPushButton("Save planned path as route")
        save_planned_route_button.clicked.connect(self.save_planned_route)
        export_route_button = QPushButton("Export selected route JSON")
        export_route_button.clicked.connect(self.export_selected_route)
        route_layout.addWidget(self.route_list)
        route_layout.addWidget(start_record_button)
        route_layout.addWidget(stop_record_button)
        route_layout.addWidget(save_planned_route_button)
        route_layout.addWidget(export_route_button)
        left_layout.addWidget(route_group)

        ros_group = QGroupBox("ROS1 / autonomous drive")
        ros_layout = QVBoxLayout(ros_group)
        publish_goal_button = QPushButton("Publish goal to /move_base_simple/goal")
        publish_goal_button.clicked.connect(self.publish_goal)
        publish_path_button = QPushButton("Publish path to /plant_nav_gui/path")
        publish_path_button.clicked.connect(self.publish_path)
        start_auto_drive_button = QPushButton("Start autonomous driving")
        start_auto_drive_button.clicked.connect(self.start_autonomous_drive)
        stop_auto_drive_button = QPushButton("Stop autonomous driving")
        stop_auto_drive_button.clicked.connect(self.stop_autonomous_drive)
        ros_layout.addWidget(publish_goal_button)
        ros_layout.addWidget(publish_path_button)
        ros_layout.addWidget(start_auto_drive_button)
        ros_layout.addWidget(stop_auto_drive_button)
        left_layout.addWidget(ros_group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Runtime log")
        left_layout.addWidget(self.log_view, 1)
        left_layout.addStretch(1)

        self.canvas = MapCanvas(self)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self.canvas)
        splitter.setSizes([420, 960])

    def log(self, text: str) -> None:
        self.log_view.append(text)

    def set_state(self, state: MapState) -> None:
        self.state = state
        self.canvas.set_state(state)
        self.map_name_label.setText(state.name)
        self.source_label.setText(f"{state.source.kind}: {state.source.path}")
        self.refresh_route_list()
        self.log(f"Loaded map: {state.name}")

    def refresh_route_list(self) -> None:
        self.route_list.clear()
        if self.state is None:
            return
        for route in self.state.routes:
            self.route_list.addItem(f"{route.name} ({len(route.points)} pts)")

    def import_ros_map(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select ROS map YAML", str(ROOT_DIR), "YAML (*.yaml *.yml)")
        if not path:
            return
        try:
            self.set_state(load_ros_map(path))
        except Exception as exc:
            QMessageBox.critical(self, "Import failed", str(exc))

    def import_pcd(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select PCD file", str(ROOT_DIR), "PCD (*.pcd)")
        if not path:
            return
        resolution, ok = QInputDialog.getDouble(self, "PCD resolution", "Voxel resolution (m, relaxed import mode)", 0.20, 0.05, 1.00, 2)
        if not ok:
            return
        self._run_import_task(
            f"Importing PCD: {Path(path).name}",
            lambda: load_ascii_pcd_projection(
                path,
                resolution=resolution,
                voxel_downsample_m=0.10,
                min_points_per_voxel=1,
                min_cluster_voxels=1,
            ),
        )

    def import_world(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Gazebo scene", str(ROOT_DIR), "World/SDF (*.world *.sdf *.xml)")
        if not path:
            return
        resolution, ok = QInputDialog.getDouble(self, "Scene resolution", "Projection resolution (m/cell)", 0.10, 0.02, 1.00, 2)
        if not ok:
            return
        extent, ok = QInputDialog.getDouble(self, "Scene extent", "Half-width projection range (m)", 20.0, 2.0, 200.0, 1)
        if not ok:
            return
        self._run_import_task(
            f"Importing scene: {Path(path).name}",
            lambda: load_world_projection(path, resolution, extent),
        )

    def save_package(self) -> None:
        if self.state is None:
            QMessageBox.information(self, "Info", "Load a map first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select map package directory", str(ROOT_DIR / "packages"))
        if not directory:
            return
        try:
            metadata_path = save_map_package(self.state, directory)
            self.log(f"Map package saved: {metadata_path}")
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def load_package(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select map package directory", str(ROOT_DIR / "packages"))
        if not directory:
            return
        try:
            self.set_state(load_map_package(directory))
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _run_import_task(self, label: str, loader) -> None:
        if self.import_worker is not None and self.import_worker.isRunning():
            QMessageBox.information(self, "Info", "Another map import is still running.")
            return
        self.import_worker = ImportWorker(loader, self)
        self.import_worker.finished_ok.connect(self._on_import_success)
        self.import_worker.failed.connect(self._on_import_failed)
        self.log(label)
        self.setEnabled(False)
        self.import_worker.start()

    def _on_import_success(self, state: MapState) -> None:
        self.setEnabled(True)
        self.set_state(state)
        self.log("Map import completed")
        self.import_worker = None

    def _on_import_failed(self, message: str) -> None:
        self.setEnabled(True)
        QMessageBox.critical(self, "Import failed", message)
        self.log(f"Map import failed: {message}")
        self.import_worker = None

    def set_canvas_mode(self, mode: str) -> None:
        if self.state is None:
            QMessageBox.information(self, "Info", "Load a map first.")
            return
        self.canvas.set_mode(mode)
        labels = {
            "start": "Click on the map to set the start point.",
            "goal": "Click on the map to set the goal point.",
            "zone": "Click multiple points to draw a no-go zone, then press 'Finish current zone'.",
        }
        self.log(labels.get(mode, mode))

    def on_canvas_point_set(self, kind: str) -> None:
        self.canvas.set_mode("idle")
        if self.state is None:
            return
        if kind == "start" and self.state.start:
            self.log(f"Start: ({self.state.start.x:.2f}, {self.state.start.y:.2f})")
        if kind == "goal" and self.state.goal:
            self.log(f"Goal: ({self.state.goal.x:.2f}, {self.state.goal.y:.2f})")

    def finish_zone(self) -> None:
        if self.state is None:
            return
        points = self.canvas.preview_zone[:]
        if len(points) < 3:
            QMessageBox.information(self, "Info", "A no-go zone needs at least 3 points.")
            return
        name, ok = QInputDialog.getText(self, "Zone name", "Enter the plant zone name", text=f"plant_zone_{len(self.state.zones) + 1}")
        if not ok or not name.strip():
            return
        self.state.zones.append(NoGoZone(name.strip(), points))
        self.canvas.preview_zone = []
        self.canvas.set_mode("idle")
        self.canvas.update()
        self.log(f"Added no-go zone: {name.strip()}")

    def clear_zones(self) -> None:
        if self.state is None:
            return
        self.state.zones.clear()
        self.canvas.preview_zone = []
        self.canvas.update()
        self.log("Cleared all no-go zones")

    def clear_planned_path(self) -> None:
        if self.state is None:
            return
        self.state.planned_path.clear()
        self.canvas.update()
        self.log("Cleared planned path")

    def plan_current_path(self) -> None:
        if self.state is None or self.state.start is None or self.state.goal is None:
            QMessageBox.information(self, "Info", "Set both start and goal first.")
            return
        try:
            self.state.planned_path = plan_path(self.state.occupancy, self.state.start, self.state.goal, self.state.zones)
            self.canvas.update()
            self.log(f"Path planned with {len(self.state.planned_path)} points")
        except Exception as exc:
            QMessageBox.warning(self, "Planning failed", str(exc))

    def start_recording(self) -> None:
        if not self.bridge.status().available:
            QMessageBox.information(self, "Info", self.bridge.status().message)
            return
        self.record_buffer = []
        self.recording_active = True
        self.log("Started /odom route recording")

    def stop_recording(self) -> None:
        if not self.recording_active:
            return
        self.recording_active = False
        if len(self.record_buffer) < 2:
            self.log("Recording too short, nothing saved")
            return
        if self.state is None:
            return
        name, ok = QInputDialog.getText(self, "Route name", "Save recorded route as", text=f"recorded_route_{len(self.state.routes) + 1}")
        if not ok or not name.strip():
            return
        self.state.routes.append(RouteRecord(name.strip(), self.record_buffer[:]))
        self.refresh_route_list()
        self.canvas.update()
        self.log(f"Saved recorded route: {name.strip()} ({len(self.record_buffer)} pts)")

    def save_planned_route(self) -> None:
        if self.state is None or len(self.state.planned_path) < 2:
            QMessageBox.information(self, "Info", "There is no planned path to save.")
            return
        name, ok = QInputDialog.getText(self, "Route name", "Save planned path as", text=f"planned_route_{len(self.state.routes) + 1}")
        if not ok or not name.strip():
            return
        self.state.routes.append(RouteRecord(name.strip(), self.state.planned_path[:]))
        self.refresh_route_list()
        self.canvas.update()
        self.log(f"Saved planned path as route: {name.strip()}")

    def export_selected_route(self) -> None:
        if self.state is None:
            return
        route = self._selected_route()
        if route is None:
            QMessageBox.information(self, "Info", "Select a route from the list first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export route", str(ROOT_DIR / f"{route.name}.json"), "JSON (*.json)")
        if not path:
            return
        payload = {
            "name": route.name,
            "points": [{"x": point.x, "y": point.y} for point in route.points],
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.log(f"Exported route: {path}")

    def publish_goal(self) -> None:
        if self.state is None or self.state.goal is None:
            QMessageBox.information(self, "Info", "Set a goal first.")
            return
        try:
            self.bridge.publish_goal(self.state.goal)
            self.log("Published goal to /move_base_simple/goal")
        except Exception as exc:
            QMessageBox.warning(self, "Publish failed", str(exc))

    def publish_path(self) -> None:
        if self.state is None or len(self.state.planned_path) < 2:
            QMessageBox.information(self, "Info", "There is no planned path.")
            return
        try:
            self.bridge.publish_path(self.state.planned_path)
            self.log("Published path to /plant_nav_gui/path")
        except Exception as exc:
            QMessageBox.warning(self, "Publish failed", str(exc))

    def _active_route_points(self) -> tuple[str, list[Pose2D]] | None:
        route = self._selected_route()
        if route is not None and len(route.points) >= 2:
            return route.name, route.points
        if self.state is not None and len(self.state.planned_path) >= 2:
            return "planned_path", self.state.planned_path
        return None

    def start_autonomous_drive(self) -> None:
        active = self._active_route_points()
        if active is None:
            QMessageBox.information(self, "Info", "Select a saved route or plan a path first.")
            return
        if self.auto_drive_process is not None and self.auto_drive_process.poll() is None:
            QMessageBox.information(self, "Info", "Autonomous driving is already running.")
            return
        route_name, points = active
        runtime_dir = ROOT_DIR / "runtime"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        route_path = runtime_dir / "active_route.json"
        payload = {
            "name": route_name,
            "points": [{"x": point.x, "y": point.y} for point in points],
        }
        route_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        script = ROOT_DIR / "autonomous_drive" / "path_follower.py"
        cmd = [sys.executable, str(script), "--route-json", str(route_path)]
        self.auto_drive_process = subprocess.Popen(cmd, cwd=str(ROOT_DIR))
        self.log(f"Autonomous driving started with route '{route_name}'")

    def stop_autonomous_drive(self) -> None:
        if self.auto_drive_process is None or self.auto_drive_process.poll() is not None:
            self.log("Autonomous driving is not running")
            return
        self.auto_drive_process.terminate()
        self.log("Autonomous driving stop requested")

    def closeEvent(self, event):  # noqa: N802
        if self.auto_drive_process is not None and self.auto_drive_process.poll() is None:
            self.auto_drive_process.terminate()
        if self.import_worker is not None and self.import_worker.isRunning():
            self.import_worker.quit()
            self.import_worker.wait(1000)
        super().closeEvent(event)

    def _selected_route(self) -> RouteRecord | None:
        if self.state is None:
            return None
        row = self.route_list.currentRow()
        if row < 0 or row >= len(self.state.routes):
            return None
        return self.state.routes[row]

    def _poll_robot_pose(self) -> None:
        self.ros_status_label.setText(self.bridge.status().message)
        pose = self.bridge.latest_pose()
        self.canvas.set_robot_pose(pose)
        if self.recording_active and pose and self.bridge.should_append_record(self.record_buffer[-1] if self.record_buffer else None, pose):
            self.record_buffer.append(Pose2D(pose.x, pose.y))


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
