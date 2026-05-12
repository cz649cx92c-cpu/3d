#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UGV_ROOT = ROOT.parent
LINERUN_ROOT = UGV_ROOT / "linerun"
CONTROL_ROOT = UGV_ROOT / "control"

if str(LINERUN_ROOT) not in sys.path:
    sys.path.insert(0, str(LINERUN_ROOT))
if str(CONTROL_ROOT) not in sys.path:
    sys.path.insert(0, str(CONTROL_ROOT))

from plant_row_runner import CommandSender, ControlState  # type: ignore  # noqa: E402
from fw_mini_controller import BodyCommand, IOCommand, SteeringCommand  # type: ignore  # noqa: E402


@dataclass
class Pose2D:
    x: float
    y: float
    yaw: float


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


class RouteFollower:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.pose: Pose2D | None = None
        self.pose_lock = threading.Lock()
        self.stop_requested = False
        self.goal_reached = False
        self.path = self._load_route(Path(args.route_json))
        self.sender = CommandSender(
            interface=args.interface,
            channel=args.channel,
            bitrate=args.bitrate,
            period_s=args.period_ms / 1000.0,
            initial_state=ControlState(
                body=BodyCommand(gear="4t4d", vx=0.0, vy=0.0, wz=0.0),
                steering=SteeringCommand(gear="4t4d", speed=0.0, angle=0.0),
                io=IOCommand(light_mode="auto", low_beam=False),
            ),
            startup_unlock_cycles=8,
            command_timeout_s=0.6,
        )
        self._setup_ros()

    def _setup_ros(self) -> None:
        import rospy  # type: ignore
        from nav_msgs.msg import Odometry  # type: ignore

        self.rospy = rospy
        if not rospy.core.is_initialized():
            rospy.init_node("plant_nav_path_follower", anonymous=True, disable_signals=True)
        rospy.Subscriber(self.args.odom_topic, Odometry, self._on_odom, queue_size=1)

    @staticmethod
    def _load_route(route_json: Path) -> list[tuple[float, float]]:
        payload = json.loads(route_json.read_text(encoding="utf-8"))
        points = [(float(item["x"]), float(item["y"])) for item in payload.get("points", [])]
        if len(points) < 2:
            raise RuntimeError(f"Route needs at least 2 points: {route_json}")
        return points

    def _on_odom(self, msg) -> None:
        pose = msg.pose.pose
        yaw = quat_to_yaw(
            float(pose.orientation.x),
            float(pose.orientation.y),
            float(pose.orientation.z),
            float(pose.orientation.w),
        )
        with self.pose_lock:
            self.pose = Pose2D(float(pose.position.x), float(pose.position.y), yaw)

    def _current_pose(self) -> Pose2D | None:
        with self.pose_lock:
            return self.pose

    def _nearest_index(self, pose: Pose2D) -> int:
        return min(range(len(self.path)), key=lambda idx: math.hypot(self.path[idx][0] - pose.x, self.path[idx][1] - pose.y))

    def _target_index(self, pose: Pose2D, start_idx: int) -> int:
        lookahead = max(0.05, float(self.args.lookahead_m))
        for idx in range(start_idx, len(self.path)):
            if math.hypot(self.path[idx][0] - pose.x, self.path[idx][1] - pose.y) >= lookahead:
                return idx
        return len(self.path) - 1

    def _build_body_command(self, pose: Pose2D) -> BodyCommand:
        nearest_idx = self._nearest_index(pose)
        goal_x, goal_y = self.path[-1]
        goal_dist = math.hypot(goal_x - pose.x, goal_y - pose.y)
        if goal_dist <= float(self.args.goal_tolerance_m):
            self.goal_reached = True
            return BodyCommand(gear="4t4d", vx=0.0, vy=0.0, wz=0.0)

        target_idx = self._target_index(pose, nearest_idx)
        tx, ty = self.path[target_idx]
        heading = math.atan2(ty - pose.y, tx - pose.x)
        heading_error = normalize_angle(heading - pose.yaw)
        speed = float(self.args.cruise_vx)
        if abs(heading_error) > 0.7:
            speed = min(speed, 0.06)
        elif abs(heading_error) > 0.35:
            speed = min(speed, 0.09)

        wz = float(self.args.k_heading) * heading_error
        wz = max(-float(self.args.max_wz), min(float(self.args.max_wz), wz))
        return BodyCommand(gear="4t4d", vx=speed, vy=0.0, wz=wz)

    def run(self) -> int:
        def _handle_stop(signum, frame):
            del signum, frame
            self.stop_requested = True

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        self.sender.start()
        self.sender.request_unlock()
        rate = self.rospy.Rate(max(1.0, 1000.0 / float(self.args.period_ms)))
        last_pose_log = 0.0

        try:
            while not self.rospy.is_shutdown() and not self.stop_requested:
                pose = self._current_pose()
                if pose is None:
                    self.sender.update(
                        BodyCommand(gear="4t4d", vx=0.0, vy=0.0, wz=0.0),
                        SteeringCommand(gear="4t4d", speed=0.0, angle=0.0),
                        IOCommand(light_mode="auto", low_beam=False),
                    )
                    rate.sleep()
                    continue

                body_cmd = self._build_body_command(pose)
                if abs(body_cmd.vx) > 1e-6 or abs(body_cmd.wz) > 1e-6:
                    self.sender.request_unlock()
                self.sender.update(
                    body_cmd,
                    SteeringCommand(gear="4t4d", speed=0.0, angle=0.0),
                    IOCommand(light_mode="auto", low_beam=False),
                )

                now = time.monotonic()
                if now - last_pose_log >= 1.0:
                    print(
                        f"[path_follower] pose=({pose.x:.2f}, {pose.y:.2f}) yaw={math.degrees(pose.yaw):.1f} "
                        f"cmd_vx={body_cmd.vx:.3f} cmd_wz={body_cmd.wz:.3f}",
                        flush=True,
                    )
                    last_pose_log = now

                if self.goal_reached:
                    print("[path_follower] goal reached, stopping vehicle", flush=True)
                    break

                rate.sleep()
        finally:
            self.sender.stop()
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="3d_nav autonomous route follower")
    parser.add_argument("--route-json", required=True)
    parser.add_argument("--odom-topic", default="/odom")
    parser.add_argument("--interface", default="socketcan")
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=500000)
    parser.add_argument("--period-ms", type=int, default=20)
    parser.add_argument("--lookahead-m", type=float, default=0.45)
    parser.add_argument("--goal-tolerance-m", type=float, default=0.25)
    parser.add_argument("--cruise-vx", type=float, default=0.12)
    parser.add_argument("--k-heading", type=float, default=1.25)
    parser.add_argument("--max-wz", type=float, default=1.2)
    return parser.parse_args()


def main() -> int:
    return RouteFollower(parse_args()).run()


if __name__ == "__main__":
    raise SystemExit(main())
