cmake_minimum_required(VERSION 3.0.2)
project(jie_octomap)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  nav_msgs
  octomap_msgs
  rospy
  sensor_msgs
  std_msgs
  tf2_ros
  visualization_msgs
)

catkin_python_setup()

add_service_files(
  FILES
  ExportNavigationSnapshot.srv
  GetNavigationMapMeta.srv
  LoadNavigationMapPackage.srv
  SaveNavigationMapPackage.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    geometry_msgs
    message_runtime
    nav_msgs
    octomap_msgs
    rospy
    sensor_msgs
    std_msgs
    tf2_ros
    visualization_msgs
)

catkin_install_python(PROGRAMS
  scripts/map_package_manager.py
  scripts/map_viewer_gui.py
  scripts/pcd_map_import_gui.py
  scripts/pcd_to_octomap_node.py
  scripts/octomap_to_occupied_markers_node.py
  scripts/plant_nav_gui.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
