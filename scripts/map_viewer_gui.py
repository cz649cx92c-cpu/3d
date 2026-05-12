#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import rospy
import tf2_ros
import vtk
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped
from nav_msgs.msg import Path as PathMsg
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSplitter,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
except ImportError:  # pragma: no cover - compatibility with older distro VTK
    from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from jie_octomap.srv import LoadNavigationMapPackage, SaveNavigationMapPackage
from jie_octomap_ros1.core import build_cube_list_marker


def _marker_to_arrays(marker: Marker | None) -> tuple[np.ndarray, np.ndarray]:
    if marker is None:
        return np.empty((0, 3), dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)
    points = np.asarray([[p.x, p.y, p.z] for p in marker.points], dtype=np.float32)
    scale = np.asarray([marker.scale.x, marker.scale.y, marker.scale.z], dtype=np.float32)
    return points, scale


class MapViewerRosBridge:
    def __init__(self) -> None:
        self.tf_parent = rospy.get_param("~tf_parent_frame", "map")
        self.tf_child = rospy.get_param("~tf_child_frame", "base_footprint")
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.start_pub = rospy.Publisher("/start_point", PointStamped, queue_size=1, latch=True)
        self.goal_pub = rospy.Publisher("/goal_point", PointStamped, queue_size=1, latch=True)
        self.goal_pose_pub = rospy.Publisher("/goal_pose", PoseStamped, queue_size=1, latch=True)
        self.initial_pose_pub = rospy.Publisher("/initialpose", PoseWithCovarianceStamped, queue_size=1, latch=True)
        self.occupied_pub = rospy.Publisher("/octomap_occupied_markers", Marker, queue_size=1, latch=True)
        self.preblocked_pub = rospy.Publisher("/preblocked_cells_markers", Marker, queue_size=1, latch=True)
        self.traversable_pub = rospy.Publisher("/traversable_cells_markers", Marker, queue_size=1, latch=True)

        self.latest_path: list[tuple[float, float, float]] = []
        self.latest_occupied: Marker | None = None
        self.latest_preblocked: Marker | None = None
        self.latest_traversable: Marker | None = None
        self.latest_risk: tuple[np.ndarray, np.ndarray] = (np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32))
        self.dirty = True
        self.path_dirty = True

        rospy.Subscriber("/planned_path", PathMsg, self._on_path, queue_size=1)
        rospy.Subscriber("/octomap_occupied_markers", Marker, self._on_occupied, queue_size=1)
        rospy.Subscriber("/preblocked_cells_markers", Marker, self._on_preblocked, queue_size=1)
        rospy.Subscriber("/traversable_cells_markers", Marker, self._on_traversable, queue_size=1)
        rospy.Subscriber("/risk_cost_cells", PointCloud2, self._on_risk, queue_size=1)

        self.save_client = rospy.ServiceProxy("/map_package_manager/save_package", SaveNavigationMapPackage)
        self.load_client = rospy.ServiceProxy("/map_package_manager/load_package", LoadNavigationMapPackage)

    def _on_path(self, message: PathMsg) -> None:
        self.latest_path = [(pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) for pose in message.poses]
        self.path_dirty = True

    def _on_occupied(self, message: Marker) -> None:
        self.latest_occupied = message
        self.dirty = True

    def _on_preblocked(self, message: Marker) -> None:
        self.latest_preblocked = message
        self.dirty = True

    def _on_traversable(self, message: Marker) -> None:
        self.latest_traversable = message
        self.dirty = True

    def _on_risk(self, message: PointCloud2) -> None:
        records = list(point_cloud2.read_points(message, field_names=("x", "y", "z", "intensity"), skip_nans=True))
        if not records:
            self.latest_risk = (np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32))
        else:
            values = np.asarray(records, dtype=np.float32)
            self.latest_risk = (values[:, :3], values[:, 3])
        self.dirty = True

    def publish_point(self, kind: str, xyz: tuple[float, float, float]) -> None:
        message = PointStamped()
        message.header.frame_id = self.tf_parent
        message.header.stamp = rospy.Time.now()
        message.point.x = float(xyz[0])
        message.point.y = float(xyz[1])
        message.point.z = float(xyz[2])
        if kind == "start":
            self.start_pub.publish(message)
        else:
            self.goal_pub.publish(message)

    def publish_goal_pose(self, xyz: tuple[float, float, float], yaw: float = 0.0) -> None:
        message = PoseStamped()
        message.header.frame_id = self.tf_parent
        message.header.stamp = rospy.Time.now()
        message.pose.position.x = float(xyz[0])
        message.pose.position.y = float(xyz[1])
        message.pose.position.z = float(xyz[2])
        message.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.orientation.w = math.cos(yaw * 0.5)
        self.goal_pose_pub.publish(message)

    def publish_initial_pose(self, xyz: tuple[float, float, float], yaw: float = 0.0) -> None:
        message = PoseWithCovarianceStamped()
        message.header.frame_id = self.tf_parent
        message.header.stamp = rospy.Time.now()
        message.pose.pose.position.x = float(xyz[0])
        message.pose.pose.position.y = float(xyz[1])
        message.pose.pose.position.z = float(xyz[2])
        message.pose.pose.orientation.z = math.sin(yaw * 0.5)
        message.pose.pose.orientation.w = math.cos(yaw * 0.5)
        message.pose.covariance[0] = 0.25
        message.pose.covariance[7] = 0.25
        message.pose.covariance[35] = 0.0685
        self.initial_pose_pub.publish(message)

    def publish_layer(self, layer_name: str, points: np.ndarray, scale: np.ndarray) -> None:
        if layer_name == "occupied":
            marker = build_cube_list_marker(self.tf_parent, "occupied_voxels", points, scale, (0.95, 0.45, 0.15, 1.0))
            self.occupied_pub.publish(marker)
        elif layer_name == "preblocked":
            marker = build_cube_list_marker(self.tf_parent, "preblocked_cells", points, scale, (0.15, 0.35, 1.0, 1.0))
            self.preblocked_pub.publish(marker)
        elif layer_name == "traversable":
            marker = build_cube_list_marker(self.tf_parent, "traversable_cells", points, scale, (0.20, 0.95, 0.55, 0.30))
            self.traversable_pub.publish(marker)

    def save_package(self, package_path: str, overwrite: bool) -> tuple[bool, str]:
        rospy.wait_for_service("/map_package_manager/save_package", timeout=2.0)
        response = self.save_client(package_path=package_path, overwrite=overwrite)
        return bool(response.success), str(response.message)

    def load_package(self, package_path: str) -> tuple[bool, str]:
        rospy.wait_for_service("/map_package_manager/load_package", timeout=2.0)
        response = self.load_client(package_path=package_path)
        return bool(response.success), str(response.message)

    def robot_pose(self) -> tuple[tuple[float, float, float], float] | None:
        try:
            transform = self.tf_buffer.lookup_transform(self.tf_parent, self.tf_child, rospy.Time(0), rospy.Duration(0.05))
        except Exception:  # noqa: BLE001
            return None
        t = transform.transform.translation
        r = transform.transform.rotation
        yaw = math.atan2(2.0 * (r.w * r.z + r.x * r.y), 1.0 - 2.0 * (r.y * r.y + r.z * r.z))
        return ((t.x, t.y, t.z), yaw)


class MapViewerWindow(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = MapViewerRosBridge()
        self.default_package_dir = Path("/root/ugv/maps/map")
        self.layer_points = {
            "occupied": np.empty((0, 3), dtype=np.float32),
            "preblocked": np.empty((0, 3), dtype=np.float32),
            "traversable": np.empty((0, 3), dtype=np.float32),
        }
        self.layer_scale = {
            "occupied": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "preblocked": np.array([0.2, 0.2, 0.2], dtype=np.float32),
            "traversable": np.array([0.2, 0.2, 0.2], dtype=np.float32),
        }
        self.path_points: list[tuple[float, float, float]] = []
        self.pick_mode: str | None = None
        self.actors: dict[str, vtk.vtkActor] = {}
        self.start_actor: vtk.vtkActor | None = None
        self.goal_actor: vtk.vtkActor | None = None
        self.robot_actor: vtk.vtkActor | None = None
        self._data_loaded_once = False
        self._build_ui()
        self._setup_vtk()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(100)

    def _build_ui(self) -> None:
        self.setWindowTitle("ROS1 Map Viewer GUI")
        self.resize(1460, 900)
        layout = QHBoxLayout(self)
        left = QVBoxLayout()
        left.setSpacing(10)

        map_group = QGroupBox("Map Package")
        map_layout = QVBoxLayout(map_group)
        root_row = QHBoxLayout()
        self.path_edit = QLineEdit(str(self.default_package_dir))
        root_btn = QPushButton("Browse")
        root_btn.clicked.connect(self._choose_path)
        root_row.addWidget(self.path_edit, 1)
        root_row.addWidget(root_btn)
        map_layout.addLayout(root_row)
        button_row = QHBoxLayout()
        open_btn = QPushButton("Open map package")
        open_btn.clicked.connect(self._open_package)
        refresh_btn = QPushButton("Refresh map")
        refresh_btn.clicked.connect(self._refresh_layers)
        save_btn = QPushButton("Save map")
        save_btn.clicked.connect(self._save_package)
        button_row.addWidget(open_btn)
        button_row.addWidget(refresh_btn)
        button_row.addWidget(save_btn)
        map_layout.addLayout(button_row)
        left.addWidget(map_group)

        view_group = QGroupBox("View Controls")
        view_layout = QVBoxLayout(view_group)
        view_row_1 = QHBoxLayout()
        fit_btn = QPushButton("Fit map")
        fit_btn.clicked.connect(self._fit_camera_to_scene)
        top_btn = QPushButton("Top")
        top_btn.clicked.connect(lambda: self._set_camera_view("top"))
        front_btn = QPushButton("Front")
        front_btn.clicked.connect(lambda: self._set_camera_view("front"))
        side_btn = QPushButton("Side")
        side_btn.clicked.connect(lambda: self._set_camera_view("side"))
        view_row_1.addWidget(fit_btn)
        view_row_1.addWidget(top_btn)
        view_row_1.addWidget(front_btn)
        view_row_1.addWidget(side_btn)
        view_layout.addLayout(view_row_1)
        self.stats_label = QLabel("No map loaded yet")
        self.stats_label.setWordWrap(True)
        view_layout.addWidget(self.stats_label)
        help_label = QLabel(
            "Mouse: left drag rotate, middle drag pan, wheel zoom. "
            "Pick a mode first, then click voxels in the 3D view."
        )
        help_label.setWordWrap(True)
        view_layout.addWidget(help_label)
        left.addWidget(view_group)

        display_group = QGroupBox("Layer Display")
        display_layout = QVBoxLayout(display_group)
        self.checkboxes = {}
        for name, label, checked in [
            ("occupied", "Occupied", True),
            ("preblocked", "Preblocked", False),
            ("traversable", "Traversable", False),
        ]:
            checkbox = QCheckBox(label)
            checkbox.setChecked(checked)
            checkbox.toggled.connect(self._render_layers)
            display_layout.addWidget(checkbox)
            self.checkboxes[name] = checkbox
        left.addWidget(display_group)

        planning_group = QGroupBox("Planning / Navigation")
        planning_layout = QVBoxLayout(planning_group)
        for text, mode in [
            ("Pick start", "start"),
            ("Pick goal", "goal"),
            ("Pick navigation target", "navigate"),
            ("Pick current pose", "current_pose"),
        ]:
            button = QPushButton(text)
            button.clicked.connect(lambda _checked=False, m=mode: self._set_pick_mode(m))
            planning_layout.addWidget(button)
        left.addWidget(planning_group)

        edit_group = QGroupBox("Grid Editing")
        edit_layout = QVBoxLayout(edit_group)
        self.edit_enabled = QCheckBox("Enable editing")
        edit_layout.addWidget(self.edit_enabled)
        radio_row = QVBoxLayout()
        self.edit_group = QButtonGroup(self)
        for idx, (name, label) in enumerate([
            ("occupied", "occupied"),
            ("preblocked", "preblocked"),
            ("traversable", "traversable"),
            ("clear", "clear"),
        ]):
            radio = QRadioButton(label)
            if idx == 0:
                radio.setChecked(True)
            self.edit_group.addButton(radio)
            self.edit_group.setId(radio, idx)
            radio.setProperty("layer_name", name)
            radio_row.addWidget(radio)
        edit_layout.addLayout(radio_row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Brush cells"))
        self.brush_spin = QSpinBox()
        self.brush_spin.setRange(1, 15)
        self.brush_spin.setValue(1)
        size_row.addWidget(self.brush_spin)
        edit_layout.addLayout(size_row)
        left.addWidget(edit_group)

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(140)
        left.addWidget(self.log_view, 1)

        left_widget = QWidget()
        left_widget.setLayout(left)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_widget)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_scroll)
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        splitter.addWidget(self.vtk_widget)
        splitter.setSizes([360, 1100])
        layout.addWidget(splitter)

    def _setup_vtk(self) -> None:
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.07, 0.09, 0.11)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        self.interactor.AddObserver("LeftButtonPressEvent", self._on_left_click)
        self.interactor.Initialize()
        self.axes_actor = vtk.vtkAxesActor()
        self.axes_actor.SetTotalLength(1.2, 1.2, 1.2)
        self.axes_actor.AxisLabelsOn()
        self.orientation_widget = vtk.vtkOrientationMarkerWidget()
        self.orientation_widget.SetOrientationMarker(self.axes_actor)
        self.orientation_widget.SetInteractor(self.interactor)
        self.orientation_widget.SetViewport(0.0, 0.0, 0.16, 0.16)
        self.orientation_widget.EnabledOn()
        self.orientation_widget.InteractiveOff()
        self.renderer.ResetCamera()

    def log(self, text: str) -> None:
        self.log_view.append(text)

    def _choose_path(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Choose map package directory", self.path_edit.text())
        if directory:
            self.path_edit.setText(directory)

    def _open_package(self) -> None:
        directory = self.path_edit.text().strip()
        if not directory:
            return
        try:
            ok, message = self.bridge.load_package(directory)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Load failed", str(exc))
            return
        self.log(message)
        if not ok:
            QMessageBox.warning(self, "Load failed", message)
        else:
            self._data_loaded_once = False

    def _save_package(self) -> None:
        directory = self.path_edit.text().strip()
        if not directory:
            QMessageBox.information(self, "Info", "Choose a package directory first.")
            return
        try:
            ok, message = self.bridge.save_package(directory, True)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))
            return
        self.log(message)
        if not ok:
            QMessageBox.warning(self, "Save failed", message)

    def _refresh_layers(self) -> None:
        for layer_name in ("occupied", "preblocked", "traversable"):
            self.bridge.publish_layer(layer_name, self.layer_points[layer_name], self.layer_scale[layer_name])
        self.log("Republished current layers")
        self._data_loaded_once = False

    def _set_pick_mode(self, mode: str) -> None:
        self.pick_mode = mode
        self.log(f"Pick mode: {mode}")

    def _tick(self) -> None:
        if self.bridge.dirty:
            occupied_points, occupied_scale = _marker_to_arrays(self.bridge.latest_occupied)
            preblocked_points, preblocked_scale = _marker_to_arrays(self.bridge.latest_preblocked)
            traversable_points, traversable_scale = _marker_to_arrays(self.bridge.latest_traversable)
            self.layer_points["occupied"] = occupied_points
            self.layer_points["preblocked"] = preblocked_points
            self.layer_points["traversable"] = traversable_points
            self.layer_scale["occupied"] = occupied_scale
            self.layer_scale["preblocked"] = preblocked_scale
            self.layer_scale["traversable"] = traversable_scale
            self.bridge.dirty = False
            self._render_layers()
            self._update_stats()
        if self.bridge.path_dirty:
            self.path_points = list(self.bridge.latest_path)
            self.bridge.path_dirty = False
            self._render_path()
        robot_pose = self.bridge.robot_pose()
        if robot_pose is not None:
            self._render_robot(robot_pose[0])

    def _render_layers(self) -> None:
        loaded_any = False
        for layer_name in ("occupied", "preblocked", "traversable"):
            actor = self.actors.get(layer_name)
            if actor is not None:
                self.renderer.RemoveActor(actor)
            points = self.layer_points[layer_name]
            if points.size == 0 or not self.checkboxes[layer_name].isChecked():
                self.actors[layer_name] = None
                continue
            loaded_any = True
            color = {
                "occupied": (0.95, 0.45, 0.15),
                "preblocked": (0.15, 0.35, 1.0),
                "traversable": (0.20, 0.95, 0.55),
            }[layer_name]
            opacity = {"occupied": 1.0, "preblocked": 1.0, "traversable": 0.30}[layer_name]
            actor = self._build_cube_actor(points, self.layer_scale[layer_name], color, opacity)
            self.actors[layer_name] = actor
            self.renderer.AddActor(actor)
        if loaded_any and not self._data_loaded_once:
            self._fit_camera_to_scene()
            self._data_loaded_once = True
        self.vtk_widget.GetRenderWindow().Render()

    def _build_cube_actor(self, points: np.ndarray, scale: np.ndarray, color: tuple[float, float, float], opacity: float) -> vtk.vtkActor:
        vtk_points = vtk.vtkPoints()
        for xyz in points:
            vtk_points.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        cube = vtk.vtkCubeSource()
        cube.SetXLength(float(scale[0]))
        cube.SetYLength(float(scale[1]))
        cube.SetZLength(float(scale[2]))
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(cube.GetOutputPort())
        glyph.SetInputData(poly)
        glyph.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        actor.GetProperty().SetOpacity(opacity)
        return actor

    def _render_path(self) -> None:
        actor = self.actors.get("path")
        if actor is not None:
            self.renderer.RemoveActor(actor)
        if len(self.path_points) < 2:
            self.actors["path"] = None
            self.vtk_widget.GetRenderWindow().Render()
            return
        vtk_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for index, xyz in enumerate(self.path_points):
            vtk_points.InsertNextPoint(float(xyz[0]), float(xyz[1]), float(xyz[2]))
            if index > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, index - 1)
                line.GetPointIds().SetId(1, index)
                lines.InsertNextCell(line)
        poly = vtk.vtkPolyData()
        poly.SetPoints(vtk_points)
        poly.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(poly)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(0.26, 0.82, 0.48)
        actor.GetProperty().SetLineWidth(4.0)
        self.actors["path"] = actor
        self.renderer.AddActor(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def _render_sphere(self, name: str, xyz: tuple[float, float, float], color: tuple[float, float, float]) -> None:
        actor = self.actors.get(name)
        if actor is not None:
            self.renderer.RemoveActor(actor)
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(float(xyz[0]), float(xyz[1]), float(xyz[2]))
        sphere.SetRadius(float(self.layer_scale["occupied"][0]) * 0.8)
        sphere.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(sphere.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*color)
        self.actors[name] = actor
        self.renderer.AddActor(actor)
        self.vtk_widget.GetRenderWindow().Render()

    def _render_robot(self, xyz: tuple[float, float, float]) -> None:
        self._render_sphere("robot", xyz, (0.0, 0.9, 1.0))

    def _scene_bounds(self) -> tuple[float, float, float, float, float, float] | None:
        layers = [points for points in self.layer_points.values() if points.size]
        if not layers:
            return None
        merged = np.vstack(layers)
        min_xyz = merged.min(axis=0)
        max_xyz = merged.max(axis=0)
        return (
            float(min_xyz[0]),
            float(max_xyz[0]),
            float(min_xyz[1]),
            float(max_xyz[1]),
            float(min_xyz[2]),
            float(max_xyz[2]),
        )

    def _fit_camera_to_scene(self) -> None:
        bounds = self._scene_bounds()
        if bounds is None:
            return
        self.renderer.ResetCamera(bounds)
        self.vtk_widget.GetRenderWindow().Render()

    def _set_camera_view(self, view_name: str) -> None:
        bounds = self._scene_bounds()
        if bounds is None:
            return
        center = (
            (bounds[0] + bounds[1]) * 0.5,
            (bounds[2] + bounds[3]) * 0.5,
            (bounds[4] + bounds[5]) * 0.5,
        )
        span = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4], 1.0)
        camera = self.renderer.GetActiveCamera()
        if view_name == "top":
            camera.SetPosition(center[0], center[1], center[2] + span * 2.2)
            camera.SetViewUp(0.0, 1.0, 0.0)
        elif view_name == "front":
            camera.SetPosition(center[0], center[1] - span * 2.2, center[2] + span * 0.4)
            camera.SetViewUp(0.0, 0.0, 1.0)
        else:
            camera.SetPosition(center[0] + span * 2.2, center[1], center[2] + span * 0.4)
            camera.SetViewUp(0.0, 0.0, 1.0)
        camera.SetFocalPoint(*center)
        self.renderer.ResetCameraClippingRange()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_stats(self) -> None:
        occupied_count = int(self.layer_points["occupied"].shape[0])
        preblocked_count = int(self.layer_points["preblocked"].shape[0])
        traversable_count = int(self.layer_points["traversable"].shape[0])
        resolution = float(self.layer_scale["occupied"][0] or 0.2)
        bounds = self._scene_bounds()
        if bounds is None:
            self.stats_label.setText("No voxel layers loaded yet")
            return
        span_x = bounds[1] - bounds[0]
        span_y = bounds[3] - bounds[2]
        span_z = bounds[5] - bounds[4]
        self.stats_label.setText(
            f"Voxel size: {resolution:.2f} m | "
            f"Occupied: {occupied_count:,} | Preblocked: {preblocked_count:,} | Traversable: {traversable_count:,}\n"
            f"Bounds: {span_x:.2f} x {span_y:.2f} x {span_z:.2f} m"
        )

    def _on_left_click(self, _obj, _event) -> None:
        click_pos = self.interactor.GetEventPosition()
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(0.0005)
        picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        if picker.GetCellId() < 0:
            self.interactor.OnLeftButtonDown()
            return
        xyz = picker.GetPickPosition()
        snapped = self._snap_point(np.asarray(xyz, dtype=np.float32))
        if self.edit_enabled.isChecked():
            self._apply_edit(snapped)
        elif self.pick_mode == "start":
            self.bridge.publish_point("start", tuple(snapped))
            self._render_sphere("start", tuple(snapped), (0.30, 0.85, 0.35))
            self.pick_mode = None
        elif self.pick_mode == "goal":
            self.bridge.publish_point("goal", tuple(snapped))
            self._render_sphere("goal", tuple(snapped), (0.95, 0.25, 0.25))
            self.pick_mode = None
        elif self.pick_mode == "navigate":
            self.bridge.publish_goal_pose(tuple(snapped))
            self._render_sphere("goal_pose", tuple(snapped), (1.0, 0.8, 0.2))
            self.pick_mode = None
        elif self.pick_mode == "current_pose":
            self.bridge.publish_initial_pose(tuple(snapped))
            self._render_sphere("initial_pose", tuple(snapped), (0.75, 0.75, 1.0))
            self.pick_mode = None
        self.interactor.OnLeftButtonDown()

    def _snap_point(self, xyz: np.ndarray) -> np.ndarray:
        resolution = float(self.layer_scale["occupied"][0] or 0.2)
        return (np.floor(xyz / resolution) + 0.5) * resolution

    def _selected_edit_layer(self) -> str:
        for button in self.edit_group.buttons():
            if button.isChecked():
                return str(button.property("layer_name"))
        return "occupied"

    def _apply_edit(self, center: np.ndarray) -> None:
        resolution = float(self.layer_scale["occupied"][0] or 0.2)
        brush = int(self.brush_spin.value())
        edits = []
        for dx in range(-(brush - 1), brush):
            for dy in range(-(brush - 1), brush):
                edits.append(center + np.array([dx * resolution, dy * resolution, 0.0], dtype=np.float32))
        target = self._selected_edit_layer()
        if target == "clear":
            for layer_name in ("occupied", "preblocked", "traversable"):
                self.layer_points[layer_name] = self._remove_points(self.layer_points[layer_name], edits, resolution)
                self.bridge.publish_layer(layer_name, self.layer_points[layer_name], self.layer_scale[layer_name])
        else:
            for layer_name in ("occupied", "preblocked", "traversable"):
                if layer_name != target:
                    self.layer_points[layer_name] = self._remove_points(self.layer_points[layer_name], edits, resolution)
            self.layer_points[target] = self._add_points(self.layer_points[target], edits, resolution)
            for layer_name in ("occupied", "preblocked", "traversable"):
                self.bridge.publish_layer(layer_name, self.layer_points[layer_name], self.layer_scale[layer_name])
        self._render_layers()

    def _remove_points(self, current: np.ndarray, edits: list[np.ndarray], resolution: float) -> np.ndarray:
        if current.size == 0:
            return current
        edit_keys = {tuple(np.round(edit / resolution).astype(int).tolist()) for edit in edits}
        kept = [point for point in current if tuple(np.round(point / resolution).astype(int).tolist()) not in edit_keys]
        if not kept:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(kept, dtype=np.float32)

    def _add_points(self, current: np.ndarray, edits: list[np.ndarray], resolution: float) -> np.ndarray:
        existing = {tuple(np.round(point / resolution).astype(int).tolist()): point for point in current} if current.size else {}
        for edit in edits:
            existing[tuple(np.round(edit / resolution).astype(int).tolist())] = edit
        return np.asarray(list(existing.values()), dtype=np.float32)


def main() -> None:
    rospy.init_node("map_viewer_gui", disable_signals=True)
    app = QApplication(sys.argv)
    window = MapViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
