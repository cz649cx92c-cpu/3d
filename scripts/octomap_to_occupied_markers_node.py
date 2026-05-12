#!/usr/bin/env python3
from __future__ import annotations

import copy

import rospy
from visualization_msgs.msg import Marker


class OctomapToOccupiedMarkersNode:
    def __init__(self) -> None:
        input_topic = rospy.get_param("~input_marker_topic", "/octomap_occupied_markers_raw")
        output_topic = rospy.get_param("~marker_topic", "/octomap_occupied_markers")
        self.publisher = rospy.Publisher(output_topic, Marker, queue_size=1, latch=True)
        self.subscriber = rospy.Subscriber(input_topic, Marker, self._on_marker, queue_size=1)

    def _on_marker(self, marker: Marker) -> None:
        outgoing = copy.deepcopy(marker)
        outgoing.ns = "occupied_voxels"
        self.publisher.publish(outgoing)


def main() -> None:
    rospy.init_node("octomap_to_occupied_markers")
    OctomapToOccupiedMarkersNode()
    rospy.spin()


if __name__ == "__main__":
    main()
