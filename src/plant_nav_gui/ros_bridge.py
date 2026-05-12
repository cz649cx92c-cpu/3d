from __future__ import annotations

import math
from dataclasses import dataclass

from .map_model import Pose2D


@dataclass
class ROSStatus:
    available: bool
    message: str


class ROSBridge:
    def __init__(self) -> None:
        self._rospy = None
        self._goal_pub = None
        self._path_pub = None
        self._latest_pose: Pose2D | None = None
        self._status = ROSStatus(False, "ROS not connected")

        try:
            import rospy
            from geometry_msgs.msg import PoseStamped
            from nav_msgs.msg import Odometry, Path

            self._rospy = rospy
            if not rospy.core.is_initialized():
                rospy.init_node("plant_nav_gui", anonymous=True, disable_signals=True)
            self._PoseStamped = PoseStamped
            self._Path = Path
            self._goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1)
            self._path_pub = rospy.Publisher("/plant_nav_gui/path", Path, queue_size=1, latch=True)
            rospy.Subscriber("/odom", Odometry, self._odom_callback, queue_size=1)
            self._status = ROSStatus(True, "ROS connected, listening on /odom")
        except Exception as exc:  # pragma: no cover - runtime ROS environment only
            self._status = ROSStatus(False, f"ROS unavailable: {exc}")

    def _odom_callback(self, msg) -> None:  # pragma: no cover - runtime ROS environment only
        self._latest_pose = Pose2D(float(msg.pose.pose.position.x), float(msg.pose.pose.position.y))

    def status(self) -> ROSStatus:
        return self._status

    def latest_pose(self) -> Pose2D | None:
        return self._latest_pose

    def publish_goal(self, point: Pose2D, frame_id: str = "map") -> None:
        if not self._status.available:
            raise RuntimeError(self._status.message)
        msg = self._PoseStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = self._rospy.Time.now()
        msg.pose.position.x = point.x
        msg.pose.position.y = point.y
        msg.pose.position.z = 0.0
        msg.pose.orientation.w = 1.0
        self._goal_pub.publish(msg)

    def publish_path(self, points: list[Pose2D], frame_id: str = "map") -> None:
        if not self._status.available:
            raise RuntimeError(self._status.message)
        path = self._Path()
        path.header.frame_id = frame_id
        path.header.stamp = self._rospy.Time.now()
        for point in points:
            pose = self._PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = point.x
            pose.pose.position.y = point.y
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

    @staticmethod
    def should_append_record(last_point: Pose2D | None, new_point: Pose2D, min_distance: float = 0.15) -> bool:
        if last_point is None:
            return True
        return math.hypot(new_point.x - last_point.x, new_point.y - last_point.y) >= min_distance
