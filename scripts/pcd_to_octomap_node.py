#!/usr/bin/env python3
from __future__ import annotations

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

    def _on_command(self, message: String) -> None:
        path = message.data.strip()
        if not path:
            return
        rospy.loginfo("pcd_to_octomap_node loading %s", path)
        try:
            points = read_ascii_pcd_points(path)
            occupied = voxelize_points(
                points,
                resolution=self.resolution,
                voxel_downsample_m=self.voxel_downsample_m,
                min_points_per_voxel=self.min_points_per_voxel,
                min_cluster_voxels=self.min_cluster_voxels,
            )
            if occupied.size == 0:
                raise ValueError("No occupied voxels remained after point filtering.")
            scale = [self.resolution, self.resolution, self.resolution]
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
            octomap = build_placeholder_octomap(self.frame_id, self.resolution, occupied)
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
