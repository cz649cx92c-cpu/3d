# jie_octomap ROS1 Port

这是把 `6-robot/jie_3d_nav` 的 `jie_octomap` 工作流往 ROS1 方向做的本地适配版，目标是尽量保留它原来的两步式流程：

1. `PCD -> OctoMap import pipeline`
2. `Map package manager + map_viewer_gui`

当前仓库里已经补上的核心入口：

- `launch/import_pcd_map.launch`
- `launch/map_manager.launch`
- `scripts/pcd_to_octomap_node.py`
- `scripts/octomap_to_occupied_markers_node.py`
- `scripts/map_package_manager.py`
- `scripts/pcd_map_import_gui.py`
- `scripts/map_viewer_gui.py`
- `srv/SaveNavigationMapPackage.srv`
- `srv/LoadNavigationMapPackage.srv`

## Build

把这个目录放进你的 catkin 工作区 `src` 后编译：

```bash
cd ~/catkin_ws
catkin_make
source devel/setup.bash
```

如果你的工作区不是这个名字，就把路径换成你自己的。

## Startup

先起导入链：

```bash
roslaunch jie_octomap import_pcd_map.launch
```

导入链现在默认带有体素安全上限：

```bash
max_occupied_voxels:=300000
```

如果终端提示体素数超过安全上限，不要继续硬跑，优先把参数调粗一些，例如：

```bash
roslaunch jie_octomap import_pcd_map.launch \
  resolution:=1.0 \
  voxel_downsample_m:=0.5 \
  min_points_per_voxel:=2 \
  min_cluster_voxels:=20
```

如果房间最外围墙壳太挡视线，可以额外剥掉最外层边界体素：

```bash
roslaunch jie_octomap import_pcd_map.launch \
  resolution:=0.5 \
  voxel_downsample_m:=0.3 \
  min_points_per_voxel:=1 \
  min_cluster_voxels:=1 \
  outer_shell_layers:=1
```

把 `outer_shell_layers` 调成 `2` 或 `3`，就会继续向里多剥几层。

这一步会启动：

- `pcd_to_octomap`
- `octomap_to_occupied_markers`
- `map_package_manager`
- `pcd_map_import_gui`

然后单独起地图管理和 3D 查看：

```bash
roslaunch jie_octomap map_manager.launch
```

这一步会启动：

- `map_package_manager`
- `map_viewer_gui`

默认地图包目录建议使用：

```bash
/root/ugv/maps/map
```

注意这里填的是“具体地图包目录”，不是父目录。
也就是说：

- 对：`/root/ugv/maps/map`
- 不对：`/root/ugv/maps`

因为 `Open map package` 会直接在该目录下找 `meta.yaml`。

## Current Scope

已经对齐的方向：

- ROS1 下保留上游的 `PCD 导入 -> 地图包 -> 3D GUI` 结构
- 地图包保存/加载
- 3D 图层显示
- 起点/终点/导航目标选择
- 栅格编辑：`occupied / preblocked / traversable / clear`

当前实现还需要你知道的点：

- 这是 ROS1 适配版，不是原版 ROS2 代码直接运行
- `Octomap` 这层目前是本地兼容实现，接口和流程尽量贴近上游，但还不是完整 ROS2 原版算法链
- `PCD` 目前优先支持 `ASCII PCD`

## Notes

当前仓库里还保留了之前的 `plant_nav_gui.py` 试验版本，但后续建议你优先走现在这套 `jie_octomap` 风格入口，不要再把它当主入口。
