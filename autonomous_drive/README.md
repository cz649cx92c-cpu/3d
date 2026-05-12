# autonomous_drive

This small project lets `3d_nav` drive the FW-mini chassis along a selected route
without opening the old `control` GUI.

It follows a saved route or planned path using:

- ROS1 `/odom` for vehicle pose
- direct CAN sending through the same FW-mini command path used by `linerun`

Main entry:

- `path_follower.py`

Example:

```bash
/usr/bin/python3 /root/ugv/3d_nav/autonomous_drive/path_follower.py \
  --route-json /root/ugv/3d_nav/runtime/last_route.json
```
