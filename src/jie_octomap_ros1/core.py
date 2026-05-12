from __future__ import annotations

import collections
import math
from pathlib import Path

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import Point
from octomap_msgs.msg import Octomap
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker


def read_ascii_pcd_points(pcd_path: str | Path) -> np.ndarray:
    pcd_path = Path(pcd_path).expanduser().resolve()
    header_done = False
    points: list[tuple[float, float, float]] = []
    with pcd_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if not header_done:
                if line.upper().startswith("DATA"):
                    if "ascii" not in line.lower():
                        raise ValueError("Only ASCII PCD is supported right now. Convert binary PCD first.")
                    header_done = True
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            points.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if not points:
        raise ValueError("No usable points were found in the PCD file.")
    return np.asarray(points, dtype=np.float32)


def voxelize_points(
    points: np.ndarray,
    resolution: float,
    voxel_downsample_m: float,
    min_points_per_voxel: int,
    min_cluster_voxels: int,
) -> np.ndarray:
    if points.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    working = points
    if voxel_downsample_m > 0.0:
        buckets: dict[tuple[int, int, int], np.ndarray] = {}
        for point in working:
            key = tuple(int(math.floor(float(value) / voxel_downsample_m)) for value in point)
            if key not in buckets:
                buckets[key] = point
        working = np.asarray(list(buckets.values()), dtype=np.float32)

    voxel_counts: dict[tuple[int, int, int], int] = collections.defaultdict(int)
    for point in working:
        key = tuple(int(math.floor(float(value) / resolution)) for value in point)
        voxel_counts[key] += 1

    occupied = {key for key, count in voxel_counts.items() if count >= max(1, int(min_points_per_voxel))}
    if not occupied:
        return np.empty((0, 3), dtype=np.float32)

    if min_cluster_voxels > 1:
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        visited: set[tuple[int, int, int]] = set()
        filtered: set[tuple[int, int, int]] = set()
        for seed in list(occupied):
            if seed in visited:
                continue
            queue = collections.deque([seed])
            visited.add(seed)
            cluster: list[tuple[int, int, int]] = []
            while queue:
                current = queue.popleft()
                cluster.append(current)
                for dx, dy, dz in neighbors:
                    nxt = (current[0] + dx, current[1] + dy, current[2] + dz)
                    if nxt in occupied and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(cluster) >= int(min_cluster_voxels):
                filtered.update(cluster)
        occupied = filtered

    if not occupied:
        return np.empty((0, 3), dtype=np.float32)

    centers = [
        ((vx + 0.5) * resolution, (vy + 0.5) * resolution, (vz + 0.5) * resolution)
        for vx, vy, vz in occupied
    ]
    return np.asarray(centers, dtype=np.float32)


def build_cube_list_marker(
    frame_id: str,
    namespace: str,
    points: np.ndarray,
    scale: np.ndarray,
    color: tuple[float, float, float, float],
) -> Marker:
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = rospy.Time.now()
    marker.ns = namespace
    marker.id = 0
    marker.type = Marker.CUBE_LIST
    marker.action = Marker.ADD
    marker.pose.orientation.w = 1.0
    marker.scale.x = float(scale[0])
    marker.scale.y = float(scale[1])
    marker.scale.z = float(scale[2])
    marker.color.r = float(color[0])
    marker.color.g = float(color[1])
    marker.color.b = float(color[2])
    marker.color.a = float(color[3])
    for xyz in points:
        point = Point()
        point.x = float(xyz[0])
        point.y = float(xyz[1])
        point.z = float(xyz[2])
        marker.points.append(point)
    return marker


def marker_to_points(marker: Marker | None) -> tuple[np.ndarray, np.ndarray]:
    if marker is None:
        return np.empty((0, 3), dtype=np.float32), np.array([0.2, 0.2, 0.2], dtype=np.float32)
    points = np.asarray([[point.x, point.y, point.z] for point in marker.points], dtype=np.float32)
    scale = np.asarray([marker.scale.x, marker.scale.y, marker.scale.z], dtype=np.float32)
    return points, scale


def build_risk_cloud(frame_id: str, points: np.ndarray, intensity: np.ndarray) -> PointCloud2:
    header = Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()
    return point_cloud2.create_cloud(
        header,
        [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
        ],
        [
            (float(point[0]), float(point[1]), float(point[2]), float(cost))
            for point, cost in zip(points, intensity)
        ],
    )


def risk_cloud_to_arrays(cloud: PointCloud2 | None) -> tuple[np.ndarray, np.ndarray]:
    if cloud is None:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    records = list(point_cloud2.read_points(cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True))
    if not records:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32)
    values = np.asarray(records, dtype=np.float32)
    return values[:, :3], values[:, 3]


def build_placeholder_octomap(frame_id: str, resolution: float, occupied_points: np.ndarray) -> Octomap:
    message = Octomap()
    message.header.frame_id = frame_id
    message.header.stamp = rospy.Time.now()
    message.binary = True
    message.id = "ColorOcTree"
    message.resolution = float(resolution)
    packed = []
    for point in occupied_points:
        packed.extend(
            [
                int(max(-128, min(127, round(float(point[0]) / resolution)))),
                int(max(-128, min(127, round(float(point[1]) / resolution)))),
                int(max(-128, min(127, round(float(point[2]) / resolution)))),
                1,
            ]
        )
    message.data = packed
    return message


def compute_bounds(*layers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = [layer for layer in layers if layer is not None and layer.size]
    if not valid:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    stacked = np.vstack(valid)
    return stacked.min(axis=0), stacked.max(axis=0)


def save_map_package(
    package_dir: str | Path,
    map_id: str,
    frame_id: str,
    resolution: float,
    octomap_msg: Octomap,
    occupied: tuple[np.ndarray, np.ndarray],
    preblocked: tuple[np.ndarray, np.ndarray],
    traversable: tuple[np.ndarray, np.ndarray],
    risk: tuple[np.ndarray, np.ndarray],
) -> Path:
    package_dir = Path(package_dir).expanduser().resolve()
    package_dir.mkdir(parents=True, exist_ok=True)
    occupied_points, occupied_scale = occupied
    preblocked_points, preblocked_scale = preblocked
    traversable_points, traversable_scale = traversable
    risk_points, risk_intensity = risk
    octomap_file = package_dir / "octomap_msg.npz"
    layers_file = package_dir / "layers.npz"
    meta_file = package_dir / "meta.yaml"

    np.savez_compressed(
        octomap_file,
        binary=np.asarray([octomap_msg.binary], dtype=np.bool_),
        octomap_id=np.asarray([octomap_msg.id]),
        resolution=np.asarray([octomap_msg.resolution], dtype=np.float64),
        frame_id=np.asarray([octomap_msg.header.frame_id]),
        data=np.asarray(octomap_msg.data, dtype=np.int16),
    )
    np.savez_compressed(
        layers_file,
        occupied_points=occupied_points,
        occupied_scale=occupied_scale.astype(np.float64),
        occupied_frame_id=np.asarray([frame_id]),
        preblocked_points=preblocked_points,
        preblocked_scale=preblocked_scale.astype(np.float64),
        preblocked_frame_id=np.asarray([frame_id]),
        traversable_points=traversable_points,
        traversable_scale=traversable_scale.astype(np.float64),
        traversable_frame_id=np.asarray([frame_id]),
        risk_points=risk_points,
        risk_intensity=risk_intensity,
        risk_frame_id=np.asarray([frame_id]),
    )
    min_bound, max_bound = compute_bounds(occupied_points, preblocked_points, traversable_points, risk_points)
    meta = {
        "map_id": map_id,
        "frame_id": frame_id,
        "resolution": float(resolution),
        "octomap_file": octomap_file.name,
        "layers_file": layers_file.name,
        "snapshot_stamp": {"sec": int(rospy.Time.now().secs), "nanosec": int(rospy.Time.now().nsecs)},
        "bounds": {"min": min_bound.tolist(), "max": max_bound.tolist()},
        "planner": {
            "robot_radius": 0.25,
            "snap_search_radius_cells": 12,
            "require_ground_support": True,
            "strict_direct_ground_support": False,
            "ground_support_xy_radius_cells": 1,
            "ground_support_depth_cells": 1,
            "enable_preblocked_costmap": True,
            "preblocked_costmap_radius_cells": 3,
            "preblocked_costmap_weight": 2.5,
        },
        "layers": {
            "occupied_count": int(occupied_points.shape[0]),
            "preblocked_count": int(preblocked_points.shape[0]),
            "traversable_count": int(traversable_points.shape[0]),
            "risk_cost_count": int(risk_points.shape[0]),
        },
    }
    with meta_file.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False, allow_unicode=True)
    return meta_file


def load_map_package(package_dir: str | Path) -> dict[str, object]:
    package_dir = Path(package_dir).expanduser().resolve()
    meta_file = package_dir / "meta.yaml"
    with meta_file.open("r", encoding="utf-8") as handle:
        meta = yaml.safe_load(handle)
    octomap_npz = np.load(package_dir / meta["octomap_file"], allow_pickle=False)
    layers_npz = np.load(package_dir / meta["layers_file"], allow_pickle=False)

    octomap_msg = Octomap()
    octomap_msg.header.frame_id = str(octomap_npz["frame_id"][0])
    octomap_msg.header.stamp = rospy.Time.now()
    octomap_msg.binary = bool(octomap_npz["binary"][0])
    octomap_msg.id = str(octomap_npz["octomap_id"][0])
    octomap_msg.resolution = float(octomap_npz["resolution"][0])
    octomap_msg.data = octomap_npz["data"].astype(np.int8).tolist()

    risk_points = layers_npz["risk_points"]
    risk_intensity = layers_npz["risk_intensity"]
    return {
        "meta": meta,
        "octomap": octomap_msg,
        "occupied_marker": build_cube_list_marker(
            str(layers_npz["occupied_frame_id"][0]),
            "occupied_voxels",
            layers_npz["occupied_points"],
            layers_npz["occupied_scale"],
            (0.95, 0.45, 0.15, 1.0),
        ),
        "preblocked_marker": build_cube_list_marker(
            str(layers_npz["preblocked_frame_id"][0]),
            "preblocked_cells",
            layers_npz["preblocked_points"],
            layers_npz["preblocked_scale"],
            (0.15, 0.35, 1.0, 1.0),
        ),
        "traversable_marker": build_cube_list_marker(
            str(layers_npz["traversable_frame_id"][0]),
            "traversable_cells",
            layers_npz["traversable_points"],
            layers_npz["traversable_scale"],
            (0.20, 0.95, 0.55, 0.30),
        ),
        "risk_cloud": build_risk_cloud(
            str(layers_npz["risk_frame_id"][0]),
            risk_points,
            risk_intensity,
        ),
    }
