from __future__ import annotations

import collections
import heapq
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


@dataclass
class Pose2D:
    x: float
    y: float


@dataclass
class NoGoZone:
    name: str
    points: list[Pose2D]


@dataclass
class RouteRecord:
    name: str
    points: list[Pose2D]


@dataclass
class MapSource:
    kind: str
    path: str
    notes: str = ""


@dataclass
class MapState:
    name: str
    occupancy: "OccupancyMap"
    source: MapSource
    start: Pose2D | None = None
    goal: Pose2D | None = None
    zones: list[NoGoZone] = field(default_factory=list)
    routes: list[RouteRecord] = field(default_factory=list)
    planned_path: list[Pose2D] = field(default_factory=list)
    octomap_path: str = ""


class OccupancyMap:
    def __init__(self, width: int, height: int, resolution: float, origin_x: float, origin_y: float, grid: list[list[int]]):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.grid = grid

    def world_to_grid(self, x: float, y: float) -> tuple[int, int] | None:
        gx = int(math.floor((x - self.origin_x) / self.resolution))
        gy = int(math.floor((y - self.origin_y) / self.resolution))
        if gx < 0 or gy < 0 or gx >= self.width or gy >= self.height:
            return None
        return gx, gy

    def grid_to_world(self, gx: int, gy: int) -> Pose2D:
        return Pose2D(
            self.origin_x + (gx + 0.5) * self.resolution,
            self.origin_y + (gy + 0.5) * self.resolution,
        )

    def is_blocked(self, gx: int, gy: int) -> bool:
        return gx < 0 or gy < 0 or gx >= self.width or gy >= self.height or self.grid[gy][gx] >= 50

    def clone(self) -> "OccupancyMap":
        return OccupancyMap(self.width, self.height, self.resolution, self.origin_x, self.origin_y, [row[:] for row in self.grid])

    def apply_zones(self, zones: list[NoGoZone]) -> "OccupancyMap":
        cloned = self.clone()
        for zone in zones:
            rasterize_zone(cloned.grid, cloned, zone.points)
        return cloned

    def to_pil_image(self) -> Image.Image:
        image = Image.new("L", (self.width, self.height), 255)
        pixels = image.load()
        for y in range(self.height):
            for x in range(self.width):
                value = self.grid[self.height - 1 - y][x]
                pixels[x, y] = 0 if value >= 50 else 255
        return image


def pose_from_list(values: Iterable[float]) -> Pose2D:
    vals = list(values)
    return Pose2D(float(vals[0]), float(vals[1]))


def _world_points_to_grid(points: list[Pose2D], occupancy: OccupancyMap) -> list[tuple[int, int]]:
    coords = []
    for point in points:
        grid_pt = occupancy.world_to_grid(point.x, point.y)
        if grid_pt is not None:
            coords.append(grid_pt)
    return coords


def rasterize_zone(grid: list[list[int]], occupancy: OccupancyMap, points: list[Pose2D]) -> None:
    if len(points) < 3:
        return
    grid_points = _world_points_to_grid(points, occupancy)
    if len(grid_points) < 3:
        return
    xs = [p[0] for p in grid_points]
    ys = [p[1] for p in grid_points]
    min_x, max_x = max(0, min(xs)), min(occupancy.width - 1, max(xs))
    min_y, max_y = max(0, min(ys)), min(occupancy.height - 1, max(ys))
    for gy in range(min_y, max_y + 1):
        for gx in range(min_x, max_x + 1):
            world = occupancy.grid_to_world(gx, gy)
            if point_in_polygon(world.x, world.y, points):
                grid[gy][gx] = 100


def point_in_polygon(x: float, y: float, points: list[Pose2D]) -> bool:
    inside = False
    j = len(points) - 1
    for i in range(len(points)):
        xi, yi = points[i].x, points[i].y
        xj, yj = points[j].x, points[j].y
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def load_ros_map(yaml_path: str | Path) -> MapState:
    yaml_path = Path(yaml_path).expanduser().resolve()
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    image_path = (yaml_path.parent / data["image"]).resolve()
    image = Image.open(image_path).convert("L")
    width, height = image.size
    negate = int(data.get("negate", 0))
    occupied_thresh = float(data.get("occupied_thresh", 0.65))
    free_thresh = float(data.get("free_thresh", 0.196))
    origin = data.get("origin", [0.0, 0.0, 0.0])
    pixels = image.load()
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for img_y in range(height):
        for x in range(width):
            intensity = pixels[x, img_y] / 255.0
            if negate:
                occ = intensity
            else:
                occ = 1.0 - intensity
            if occ > occupied_thresh:
                value = 100
            elif occ < free_thresh:
                value = 0
            else:
                value = 100
            grid[height - 1 - img_y][x] = value
    occupancy = OccupancyMap(width, height, float(data["resolution"]), float(origin[0]), float(origin[1]), grid)
    return MapState(
        name=yaml_path.stem,
        occupancy=occupancy,
        source=MapSource("ros_map", str(yaml_path), "ROS 2D occupancy map projected into the local planning grid."),
    )


def load_ascii_pcd_projection(
    pcd_path: str | Path,
    resolution: float = 0.20,
    voxel_downsample_m: float = 0.10,
    min_points_per_voxel: int = 2,
    min_cluster_voxels: int = 2,
) -> MapState:
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

    if voxel_downsample_m > 0.0:
        downsampled: dict[tuple[int, int, int], tuple[float, float, float]] = {}
        for x, y, z in points:
            key = (
                int(math.floor(x / voxel_downsample_m)),
                int(math.floor(y / voxel_downsample_m)),
                int(math.floor(z / voxel_downsample_m)),
            )
            if key not in downsampled:
                downsampled[key] = (x, y, z)
        points = list(downsampled.values())

    voxel_counts: dict[tuple[int, int, int], int] = collections.defaultdict(int)
    for x, y, z in points:
        key = (
            int(math.floor(x / resolution)),
            int(math.floor(y / resolution)),
            int(math.floor(z / resolution)),
        )
        voxel_counts[key] += 1

    occupied_voxels = {key for key, count in voxel_counts.items() if count >= max(1, int(min_points_per_voxel))}
    if not occupied_voxels:
        raise ValueError("No occupied voxels remained after point filtering.")

    if min_cluster_voxels > 1:
        visited: set[tuple[int, int, int]] = set()
        filtered: set[tuple[int, int, int]] = set()
        neighbors = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]
        for seed in list(occupied_voxels):
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
                    if nxt in occupied_voxels and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
            if len(cluster) >= int(min_cluster_voxels):
                filtered.update(cluster)
        occupied_voxels = filtered
        if not occupied_voxels:
            raise ValueError("All occupied voxels were removed by cluster filtering.")

    xy_cells = {(vx, vy) for vx, vy, _ in occupied_voxels}
    min_x = min(cell[0] for cell in xy_cells)
    max_x = max(cell[0] for cell in xy_cells)
    min_y = min(cell[1] for cell in xy_cells)
    max_y = max(cell[1] for cell in xy_cells)
    width = max(1, (max_x - min_x) + 3)
    height = max(1, (max_y - min_y) + 3)
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for cell_x, cell_y in xy_cells:
        gx = (cell_x - min_x) + 1
        gy = (cell_y - min_y) + 1
        if 0 <= gx < width and 0 <= gy < height:
            grid[gy][gx] = 100

    occupancy = OccupancyMap(
        width,
        height,
        resolution,
        (min_x - 1) * resolution,
        (min_y - 1) * resolution,
        grid,
    )
    return MapState(
        name=pcd_path.stem,
        occupancy=occupancy,
        source=MapSource(
            "pcd_projection",
            str(pcd_path),
            "PCD processed with voxel counting and small-cluster filtering, then projected into the local planning grid.",
        ),
    )


def _parse_pose(element: ET.Element | None) -> tuple[float, float, float]:
    if element is None or not (element.text or "").strip():
        return 0.0, 0.0, 0.0
    values = [float(v) for v in element.text.split()]
    x = values[0] if len(values) > 0 else 0.0
    y = values[1] if len(values) > 1 else 0.0
    yaw = values[5] if len(values) > 5 else 0.0
    return x, y, yaw


def _rotate(x: float, y: float, yaw: float) -> tuple[float, float]:
    cy = math.cos(yaw)
    sy = math.sin(yaw)
    return x * cy - y * sy, x * sy + y * cy


def _rasterize_rotated_box(grid: list[list[int]], occupancy: OccupancyMap, center_x: float, center_y: float, size_x: float, size_y: float, yaw: float) -> None:
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    corners = []
    for dx, dy in [(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)]:
        rx, ry = _rotate(dx, dy, yaw)
        corners.append(Pose2D(center_x + rx, center_y + ry))
    rasterize_zone(grid, occupancy, corners)


def _rasterize_cylinder(grid: list[list[int]], occupancy: OccupancyMap, center_x: float, center_y: float, radius: float) -> None:
    min_pt = occupancy.world_to_grid(center_x - radius, center_y - radius)
    max_pt = occupancy.world_to_grid(center_x + radius, center_y + radius)
    if min_pt is None or max_pt is None:
        return
    min_x, min_y = min_pt
    max_x, max_y = max_pt
    for gy in range(min(min_y, max_y), max(min_y, max_y) + 1):
        for gx in range(min(min_x, max_x), max(min_x, max_x) + 1):
            world = occupancy.grid_to_world(gx, gy)
            if math.hypot(world.x - center_x, world.y - center_y) <= radius:
                grid[gy][gx] = 100


def load_world_projection(world_path: str | Path, resolution: float = 0.10, extent: float = 20.0) -> MapState:
    world_path = Path(world_path).expanduser().resolve()
    root = ET.fromstring(world_path.read_text(encoding="utf-8"))
    occupancy = OccupancyMap(
        width=int((extent * 2) / resolution),
        height=int((extent * 2) / resolution),
        resolution=resolution,
        origin_x=-extent,
        origin_y=-extent,
        grid=[[0 for _ in range(int((extent * 2) / resolution))] for _ in range(int((extent * 2) / resolution))],
    )

    for collision in root.findall(".//collision"):
        pose_x, pose_y, pose_yaw = _parse_pose(collision.find("pose"))
        geometry = collision.find("geometry")
        if geometry is None:
            continue
        box = geometry.find("box")
        cylinder = geometry.find("cylinder")
        if box is not None:
            size_text = (box.findtext("size") or "").split()
            if len(size_text) >= 2:
                _rasterize_rotated_box(
                    occupancy.grid,
                    occupancy,
                    pose_x,
                    pose_y,
                    float(size_text[0]),
                    float(size_text[1]),
                    pose_yaw,
                )
        elif cylinder is not None:
            radius = float(cylinder.findtext("radius") or "0")
            _rasterize_cylinder(occupancy.grid, occupancy, pose_x, pose_y, radius)

    return MapState(
        name=world_path.stem,
        occupancy=occupancy,
        source=MapSource("world_projection", str(world_path), "Gazebo world/sdf projected into the local planning grid using a top-down view."),
    )


def _serialize_points(points: list[Pose2D]) -> list[list[float]]:
    return [[point.x, point.y] for point in points]


def save_map_package(state: MapState, target_dir: str | Path) -> Path:
    target_dir = Path(target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / "occupancy.png"
    metadata_path = target_dir / "package.json"

    state.occupancy.to_pil_image().save(image_path)
    source_path = Path(state.source.path)
    if source_path.exists() and source_path.is_file():
        copied_name = source_path.name
        copied_path = target_dir / copied_name
        if copied_path.resolve() != source_path.resolve():
            shutil.copy2(source_path, copied_path)

    metadata = {
        "name": state.name,
        "resolution": state.occupancy.resolution,
        "origin": [state.occupancy.origin_x, state.occupancy.origin_y],
        "width": state.occupancy.width,
        "height": state.occupancy.height,
        "source": asdict(state.source),
        "start": _serialize_points([state.start])[0] if state.start else None,
        "goal": _serialize_points([state.goal])[0] if state.goal else None,
        "zones": [{"name": zone.name, "points": _serialize_points(zone.points)} for zone in state.zones],
        "routes": [{"name": route.name, "points": _serialize_points(route.points)} for route in state.routes],
        "planned_path": _serialize_points(state.planned_path),
        "octomap_path": state.octomap_path,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def load_map_package(package_dir: str | Path) -> MapState:
    package_dir = Path(package_dir).expanduser().resolve()
    metadata = json.loads((package_dir / "package.json").read_text(encoding="utf-8"))
    image = Image.open(package_dir / "occupancy.png").convert("L")
    width, height = image.size
    pixels = image.load()
    grid = [[0 for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            value = 100 if pixels[x, height - 1 - y] < 128 else 0
            grid[y][x] = value
    occupancy = OccupancyMap(
        width=width,
        height=height,
        resolution=float(metadata["resolution"]),
        origin_x=float(metadata["origin"][0]),
        origin_y=float(metadata["origin"][1]),
        grid=grid,
    )
    state = MapState(
        name=metadata["name"],
        occupancy=occupancy,
        source=MapSource(**metadata["source"]),
        start=pose_from_list(metadata["start"]) if metadata.get("start") else None,
        goal=pose_from_list(metadata["goal"]) if metadata.get("goal") else None,
        zones=[NoGoZone(item["name"], [pose_from_list(p) for p in item["points"]]) for item in metadata.get("zones", [])],
        routes=[RouteRecord(item["name"], [pose_from_list(p) for p in item["points"]]) for item in metadata.get("routes", [])],
        planned_path=[pose_from_list(p) for p in metadata.get("planned_path", [])],
        octomap_path=metadata.get("octomap_path", ""),
    )
    return state


def plan_path(occupancy: OccupancyMap, start: Pose2D, goal: Pose2D, zones: list[NoGoZone] | None = None) -> list[Pose2D]:
    working = occupancy.apply_zones(zones or [])
    start_idx = working.world_to_grid(start.x, start.y)
    goal_idx = working.world_to_grid(goal.x, goal.y)
    if start_idx is None or goal_idx is None:
        raise ValueError("The start or goal is outside the map bounds.")
    if working.is_blocked(*start_idx) or working.is_blocked(*goal_idx):
        raise ValueError("The start or goal is inside an obstacle or no-go zone.")

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    frontier: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(frontier, (0.0, start_idx))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_idx: None}
    cost_so_far: dict[tuple[int, int], float] = {start_idx: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)
        if current == goal_idx:
            break
        for dx, dy in neighbors:
            nxt = (current[0] + dx, current[1] + dy)
            if working.is_blocked(*nxt):
                continue
            step_cost = math.sqrt(2.0) if dx and dy else 1.0
            new_cost = cost_so_far[current] + step_cost
            if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                cost_so_far[nxt] = new_cost
                priority = new_cost + math.hypot(goal_idx[0] - nxt[0], goal_idx[1] - nxt[1])
                heapq.heappush(frontier, (priority, nxt))
                came_from[nxt] = current

    if goal_idx not in came_from:
        raise ValueError("No traversable path was found. Adjust the no-go zones or move the start/goal.")

    path = []
    current = goal_idx
    while current is not None:
        path.append(working.grid_to_world(*current))
        current = came_from[current]
    path.reverse()
    return simplify_path(path, occupancy.resolution * 0.5)


def simplify_path(points: list[Pose2D], tolerance: float) -> list[Pose2D]:
    if len(points) <= 2:
        return points
    simplified = [points[0]]
    for idx in range(1, len(points) - 1):
        prev_pt = simplified[-1]
        curr_pt = points[idx]
        next_pt = points[idx + 1]
        area = abs(
            prev_pt.x * (curr_pt.y - next_pt.y)
            + curr_pt.x * (next_pt.y - prev_pt.y)
            + next_pt.x * (prev_pt.y - curr_pt.y)
        ) / 2.0
        base = math.hypot(next_pt.x - prev_pt.x, next_pt.y - prev_pt.y) or 1e-9
        distance = (2.0 * area) / base
        if distance >= tolerance:
            simplified.append(curr_pt)
    simplified.append(points[-1])
    return simplified
