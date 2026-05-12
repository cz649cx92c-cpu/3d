#!/usr/bin/env python3
from __future__ import annotations

import json

import rospy
from octomap_msgs.msg import Octomap
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from jie_octomap_ros1.core import (
    build_cube_list_marker,
    build_placeholder_octomap,
    voxelize_points,
    read_ascii_pcd_points,
)


class PcdToOctomapNode:
    def __init__(self) -> None:
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.resolution = float(rospy.get_param("~resolution", 0.2))
        self.voxel_downsample_m = float(rospy.get_param("~voxel_downsample_m", 0.1))
        self.min_points_per_voxel = int(rospy.get_param("~min_points_per_voxel", 2))
        self.min_cluster_voxels = int(rospy.get_param("~min_cluster_voxels", 2))
        self.outer_shell_layers = int(rospy.get_param("~outer_shell_layers", 0))
        self.max_occupied_voxels = int(rospy.get_param("~max_occupied_voxels", 300000))
        self.pcd_cmd_topic = rospy.get_param("~pcd_file_cmd_topic", "/pcd_file_cmd")
        self.octomap_topic = rospy.get_param("~octomap_topic", "/octomap")
        self.raw_marker_topic = rospy.get_param("~raw_marker_topic", "/octomap_occupied_markers_raw")
        self.traversable_topic = rospy.get_param("~traversable_marker_topic", "/traversable_cells_markers")
        self.preblocked_topic = rospy.get_param("~preblocked_marker_topic", "/preblocked_cells_markers")

        self.octomap_pub = rospy.Publisher(self.octomap_topic, Octomap, queue_size=1, latch=True)
        self.raw_marker_pub = rospy.Publisher(self.raw_marker_topic, Marker, queue_size=1, latch=True)
        self.traversable_pub = rospy.Publisher(self.traversable_topic, Marker, queue_size=1, latch=True)
        self.preblocked_pub = rospy.Publisher(self.preblocked_topic, Marker, queue_size=1, latch=True)
        self.cmd_sub = rospy.Subscriber(self.pcd_cmd_topic, String, self._on_command, queue_size=1)

    def _command_settings(self, payload: dict[str, object]) -> dict[str, object]:
        return {
            "path": str(payload.get("path", "")).strip(),
            "resolution": float(payload.get("resolution", self.resolution)),
            "voxel_downsample_m": float(payload.get("voxel_downsample_m", self.voxel_downsample_m)),
            "min_points_per_voxel": int(payload.get("min_points_per_voxel", self.min_points_per_voxel)),
            "min_cluster_voxels": int(payload.get("min_cluster_voxels", self.min_cluster_voxels)),
            "outer_shell_layers": int(payload.get("outer_shell_layers", self.outer_shell_layers)),
            "max_occupied_voxels": int(payload.get("max_occupied_voxels", self.max_occupied_voxels)),
            "z_min": payload.get("z_min"),
            "z_max": payload.get("z_max"),
        }

    def _on_command(self, message: String) -> None:
        raw = message.data.strip()
        if not raw:
            return
        try:
            payload = json.loads(raw) if raw.startswith("{") else {"path": raw}
        except Exception:  # noqa: BLE001
            payload = {"path": raw}
        settings = self._command_settings(payload)
        path = str(settings["path"])
        if not path:
            return
        rospy.loginfo(
            "pcd_to_octomap_node loading %s | res=%.3f downsample=%.3f min_pts=%d min_cluster=%d outer_shell=%d z_min=%s z_max=%s",
            path,
            settings["resolution"],
            settings["voxel_downsample_m"],
            settings["min_points_per_voxel"],
            settings["min_cluster_voxels"],
            settings["outer_shell_layers"],
            settings["z_min"],
            settings["z_max"],
        )
        try:
            points = read_ascii_pcd_points(path)
            occupied = voxelize_points(
                points,
                resolution=float(settings["resolution"]),
                voxel_downsample_m=float(settings["voxel_downsample_m"]),
                min_points_per_voxel=int(settings["min_points_per_voxel"]),
                min_cluster_voxels=int(settings["min_cluster_voxels"]),
                outer_shell_layers=int(settings["outer_shell_layers"]),
                z_min=None if settings["z_min"] in (None, "") else float(settings["z_min"]),
                z_max=None if settings["z_max"] in (None, "") else float(settings["z_max"]),
            )
            if occupied.size == 0:
                raise ValueError("No occupied voxels remained after point filtering.")
            if len(occupied) > int(settings["max_occupied_voxels"]):
                raise ValueError(
                    "Occupied voxel count %d exceeds safety limit %d. "
                    "Increase resolution / voxel_downsample_m or raise filtering thresholds."
                    % (len(occupied), int(settings["max_occupied_voxels"]))
                )
            scale = [float(settings["resolution"])] * 3
            occupied_marker = build_cube_list_marker(
                self.frame_id,
                "occupied_voxels_raw",
                occupied,
                scale,
                (0.95, 0.45, 0.15, 1.0),
            )
            empty = occupied[:0]
            traversable_marker = build_cube_list_marker(
                self.frame_id,
                "traversable_cells",
                empty,
                scale,
                (0.20, 0.95, 0.55, 0.30),
            )
            preblocked_marker = build_cube_list_marker(
                self.frame_id,
                "preblocked_cells",
                empty,
                scale,
                (0.15, 0.35, 1.0, 1.0),
            )
            octomap = build_placeholder_octomap(self.frame_id, float(settings["resolution"]), occupied)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr("failed to import pcd %s: %s", path, exc)
            return

        self.octomap_pub.publish(octomap)
        self.raw_marker_pub.publish(occupied_marker)
        self.traversable_pub.publish(traversable_marker)
        self.preblocked_pub.publish(preblocked_marker)
        rospy.loginfo("published %d occupied voxels from %s", len(occupied_marker.points), path)


def main() -> None:
    rospy.init_node("pcd_to_octomap")
    PcdToOctomapNode()
    rospy.spin()


if __name__ == "__main__":
    main()
