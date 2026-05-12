#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import rospy
from octomap_msgs.msg import Octomap
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import Marker

from jie_octomap.srv import LoadNavigationMapPackage, LoadNavigationMapPackageResponse
from jie_octomap.srv import SaveNavigationMapPackage, SaveNavigationMapPackageResponse
from jie_octomap_ros1.core import (
    build_placeholder_octomap,
    load_map_package,
    marker_to_points,
    risk_cloud_to_arrays,
    save_map_package,
)


class MapPackageManager:
    def __init__(self) -> None:
        self.octomap_topic = rospy.get_param("~octomap_topic", "/octomap")
        self.occupied_marker_topic = rospy.get_param("~occupied_marker_topic", "/octomap_occupied_markers")
        self.preblocked_topic = rospy.get_param("~preblocked_topic", "/preblocked_cells_markers")
        self.traversable_topic = rospy.get_param("~traversable_topic", "/traversable_cells_markers")
        self.risk_topic = rospy.get_param("~risk_cost_topic", "/risk_cost_cells")
        self.autoload_package_path = rospy.get_param("~autoload_package_path", "").strip()
        self.default_resolution = float(rospy.get_param("~resolution", 0.2))

        self.latest_octomap: Octomap | None = None
        self.latest_occupied: Marker | None = None
        self.latest_preblocked: Marker | None = None
        self.latest_traversable: Marker | None = None
        self.latest_risk: PointCloud2 | None = None
        self.latest_map_id = "loaded_map"
        self.latest_frame_id = "map"

        self.octomap_pub = rospy.Publisher(self.octomap_topic, Octomap, queue_size=1, latch=True)
        self.occupied_pub = rospy.Publisher(self.occupied_marker_topic, Marker, queue_size=1, latch=True)
        self.preblocked_pub = rospy.Publisher(self.preblocked_topic, Marker, queue_size=1, latch=True)
        self.traversable_pub = rospy.Publisher(self.traversable_topic, Marker, queue_size=1, latch=True)
        self.risk_pub = rospy.Publisher(self.risk_topic, PointCloud2, queue_size=1, latch=True)

        rospy.Subscriber(self.octomap_topic, Octomap, self._on_octomap, queue_size=1)
        rospy.Subscriber(self.occupied_marker_topic, Marker, self._on_occupied, queue_size=1)
        rospy.Subscriber(self.preblocked_topic, Marker, self._on_preblocked, queue_size=1)
        rospy.Subscriber(self.traversable_topic, Marker, self._on_traversable, queue_size=1)
        rospy.Subscriber(self.risk_topic, PointCloud2, self._on_risk, queue_size=1)

        self.save_service = rospy.Service("~save_package", SaveNavigationMapPackage, self._handle_save)
        self.load_service = rospy.Service("~load_package", LoadNavigationMapPackage, self._handle_load)

        if self.autoload_package_path:
            rospy.Timer(rospy.Duration(1.0), self._autoload_once, oneshot=True)

    def _on_octomap(self, message: Octomap) -> None:
        self.latest_octomap = copy.deepcopy(message)
        self.latest_frame_id = message.header.frame_id or self.latest_frame_id

    def _on_occupied(self, message: Marker) -> None:
        if message.type == Marker.CUBE_LIST:
            self.latest_occupied = copy.deepcopy(message)
            self.latest_frame_id = message.header.frame_id or self.latest_frame_id

    def _on_preblocked(self, message: Marker) -> None:
        if message.type == Marker.CUBE_LIST:
            self.latest_preblocked = copy.deepcopy(message)
            self.latest_frame_id = message.header.frame_id or self.latest_frame_id

    def _on_traversable(self, message: Marker) -> None:
        if message.type == Marker.CUBE_LIST:
            self.latest_traversable = copy.deepcopy(message)
            self.latest_frame_id = message.header.frame_id or self.latest_frame_id

    def _on_risk(self, message: PointCloud2) -> None:
        self.latest_risk = copy.deepcopy(message)
        self.latest_frame_id = message.header.frame_id or self.latest_frame_id

    def _ensure_octomap(self) -> Octomap:
        if self.latest_octomap is not None:
            return copy.deepcopy(self.latest_octomap)
        occupied_points, _ = marker_to_points(self.latest_occupied)
        return build_placeholder_octomap(self.latest_frame_id, self.default_resolution, occupied_points)

    def _handle_save(self, request: SaveNavigationMapPackage.Request) -> SaveNavigationMapPackageResponse:
        response = SaveNavigationMapPackageResponse()
        package_dir = Path(request.package_path).expanduser()
        if package_dir.exists() and not request.overwrite:
            response.success = False
            response.message = f"package path already exists: {package_dir}"
            return response
        try:
            manifest = save_map_package(
                package_dir=package_dir,
                map_id=self.latest_map_id,
                frame_id=self.latest_frame_id,
                resolution=self.default_resolution if self.latest_octomap is None else float(self.latest_octomap.resolution),
                octomap_msg=self._ensure_octomap(),
                occupied=marker_to_points(self.latest_occupied),
                preblocked=marker_to_points(self.latest_preblocked),
                traversable=marker_to_points(self.latest_traversable),
                risk=risk_cloud_to_arrays(self.latest_risk),
            )
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
            return response
        response.success = True
        response.message = "map package saved"
        response.manifest_path = str(manifest)
        return response

    def _handle_load(self, request: LoadNavigationMapPackage.Request) -> LoadNavigationMapPackageResponse:
        response = LoadNavigationMapPackageResponse()
        try:
            loaded = load_map_package(request.package_path)
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc)
            return response
        self.latest_octomap = copy.deepcopy(loaded["octomap"])
        self.latest_occupied = copy.deepcopy(loaded["occupied_marker"])
        self.latest_preblocked = copy.deepcopy(loaded["preblocked_marker"])
        self.latest_traversable = copy.deepcopy(loaded["traversable_marker"])
        self.latest_risk = copy.deepcopy(loaded["risk_cloud"])
        self.latest_map_id = str(loaded["meta"].get("map_id", "loaded_map"))
        self.latest_frame_id = str(loaded["meta"].get("frame_id", "map"))
        self.octomap_pub.publish(self.latest_octomap)
        self.occupied_pub.publish(self.latest_occupied)
        self.preblocked_pub.publish(self.latest_preblocked)
        self.traversable_pub.publish(self.latest_traversable)
        self.risk_pub.publish(self.latest_risk)
        response.success = True
        response.message = "map package loaded"
        response.map_id = self.latest_map_id
        return response

    def _autoload_once(self, _event) -> None:
        request = LoadNavigationMapPackage.Request()
        request.package_path = self.autoload_package_path
        self._handle_load(request)


def main() -> None:
    rospy.init_node("map_package_manager")
    MapPackageManager()
    rospy.spin()


if __name__ == "__main__":
    main()
