# D1 Max 的 ROS 2 節點與 topic 全圖（由 `ros2 node info` 重建）

狗上沒有 GUI，`rqt_graph` 起不來。這份是把每個節點的 `ros2 node info` 撈回本機重建的同一張圖。兩塊板是**兩個獨立的 ROS DOMAIN**（RK=66、NX=24），只有影像 topic 由 `bridge_image_topics_*` 橋接。

---

## Orin NX（應用板）：42 個節點、223 個 topic

### 節點 → 發布 / 訂閱

| 節點 | 分類 | 發布 | 訂閱 | 服務 | 動作 |
|---|---|---|---|---|---|
| `/MEB` | 導航 | 8 | 4 | 6 | 0 |
| `/arc_lvio_node` | SLAM／定位 | 5 | 6 | 6 | 0 |
| `/arc_mapping_node` | SLAM／定位 | 4 | 3 | 7 | 0 |
| `/arc_state_machine_node` | 其他 | 15 | 15 | 6 | 0 |
| `/aritag_location_pipline` | SLAM／定位 | 7 | 4 | 6 | 0 |
| `/behavior_server` | 導航 | 5 | 5 | 11 | 4 |
| `/bridge_image_topics_24` | 其他 | 3 | 1 | 0 | 0 |
| `/bt_navigator` | 導航 | 6 | 7 | 11 | 4 |
| `/bt_navigator_navigate_through_poses_rclcpp_node` | 導航 | 9 | 1 | 10 | 6 |
| `/bt_navigator_navigate_to_pose_rclcpp_node` | 導航 | 11 | 1 | 7 | 3 |
| `/charging_alignment_server` | 導航 | 11 | 5 | 6 | 0 |
| `/collision_monitor` | 導航 | 9 | 5 | 11 | 0 |
| `/controller_server` | 導航 | 22 | 7 | 11 | 1 |
| `/error_aggregator` | 其他 | 16 | 16 | 17 | 0 |
| `/global_costmap/global_costmap` | 導航 | 10 | 5 | 16 | 0 |
| `/imu_driver` | 感測驅動 | 3 | 1 | 6 | 0 |
| `/kanon_rust_charge` | 導航 | 3 | 4 | 1 | 0 |
| `/kanon_rust_loc` | 導航 | 1 | 1 | 2 | 0 |
| `/kanon_rust_nav` | 導航 | 2 | 2 | 1 | 0 |
| `/kanon_rust_slam` | SLAM／定位 | 1 | 1 | 1 | 0 |
| `/launch_ros_2483` | 其他 | 2 | 0 | 6 | 0 |
| `/lifecycle_manager_navigation` | 導航 | 4 | 2 | 24 | 0 |
| `/local_costmap/local_costmap` | 導航 | 8 | 6 | 15 | 0 |
| `/local_costmap_controller/local_costmap_controller` | 導航 | 8 | 5 | 15 | 0 |
| `/localization` | SLAM／定位 | 12 | 12 | 9 | 0 |
| `/map_server` | 導航 | 5 | 2 | 13 | 0 |
| `/nav2_container` | 導航 | 1 | 1 | 0 | 0 |
| `/perception_jobs` | 感知 | 6 | 3 | 7 | 0 |
| `/planner_server` | 導航 | 49 | 4 | 23 | 3 |
| `/remoix_rust_interface` | 其他 | 5 | 6 | 0 | 0 |
| `/robot_slam` | SLAM／定位 | 6 | 6 | 8 | 0 |
| `/robot_tf` | SLAM／定位 | 6 | 6 | 6 | 0 |
| `/ros2_alg_interface` | 其他 | 3 | 5 | 0 | 0 |
| `/rslidar_sdk/param_handle` | 感測驅動 | 2 | 1 | 6 | 0 |
| `/rslidar_sdk/rslidar_points_destination_0` | 感測驅動 | 4 | 1 | 6 | 0 |
| `/rslidar_sdk/rslidar_points_destination_1` | 感測驅動 | 4 | 1 | 6 | 0 |
| `/rust_debug_node` | 其他 | 1 | 2 | 0 | 0 |
| `/sixents_gps_driver` | 感測驅動 | 5 | 1 | 6 | 0 |
| `/uss_driver` | 感測驅動 | 4 | 1 | 6 | 0 |
| `/uwb_driver` | 感測驅動 | 3 | 1 | 6 | 0 |
| `/velocity_optimizer` | 導航 | 6 | 3 | 11 | 0 |
| `/waypoint_follower` | 導航 | 10 | 8 | 19 | 4 |

### Topic → 發布者 / 訂閱者

| Topic | 型別 | 發布者 | 訂閱者 |
|---|---|---|---|
| `/aligned_points` | `sensor_msgs/msg/PointCloud2` | `/localization` | — |
| `/arc/arc_change_flag` | `robots_dog_msgs/msg/ArcChangeFlag` | `/arc_state_machine_node` | — |
| `/arc/arc_change_flag_test` | `robots_dog_msgs/msg/ArcChangeFlag` | — | `/arc_state_machine_node` |
| `/arc/arc_module_states_debug_info` | `std_msgs/msg/String` | `/arc_state_machine_node` | `/rust_debug_node` |
| `/arc/arc_state` | `robots_dog_msgs/msg/ArcState` | `/arc_state_machine_node` | `/aritag_location_pipline`、`/kanon_rust_charge`、`/remoix_rust_interface` |
| `/arc/arc_state_debug_info` | `std_msgs/msg/String` | `/arc_state_machine_node` | — |
| `/arc/calibration_state` | `robots_dog_msgs/msg/ArcModuleState` | `/aritag_location_pipline` | `/arc_state_machine_node` |
| `/arc/dock_pose` | `robots_dog_msgs/msg/DockPoseStamped` | `/arc_state_machine_node` | `/charging_alignment_server` |
| `/arc/dock_state` | `robots_dog_msgs/msg/DockState` | `/remoix_rust_interface` | `/arc_state_machine_node`、`/kanon_rust_charge` |
| `/arc/mc_mode_cmd` | `robots_dog_msgs/msg/McModeCmd` | `/arc_state_machine_node` | `/remoix_rust_interface` |
| `/arc/mc_state` | `robots_dog_msgs/msg/McState` | `/remoix_rust_interface` | `/arc_state_machine_node` |
| `/arc/nav_alignment/cmd_vel_raw` | `geometry_msgs/msg/Twist` | `/charging_alignment_server` | — |
| `/arc/nav_alignment/debug_info` | `std_msgs/msg/String` | `/charging_alignment_server` | — |
| `/arc/nav_alignment/error_vector` | `geometry_msgs/msg/PointStamped` | `/charging_alignment_server` | — |
| `/arc/nav_coarse_alignment_mode_cmd` | `robots_dog_msgs/msg/ArcModuleCmd` | `/arc_state_machine_node` | `/charging_alignment_server` |
| `/arc/nav_coarse_alignment_state` | `robots_dog_msgs/msg/ArcModuleState` | `/charging_alignment_server` | `/arc_state_machine_node` |
| `/arc/nav_contact_alignment_mode_cmd` | `robots_dog_msgs/msg/ArcModuleCmd` | `/arc_state_machine_node` | `/charging_alignment_server` |
| `/arc/nav_contact_alignment_state` | `robots_dog_msgs/msg/ArcModuleState` | `/charging_alignment_server` | `/arc_state_machine_node` |
| `/arc/nav_fine_alignment_mode_cmd` | `robots_dog_msgs/msg/ArcModuleCmd` | `/arc_state_machine_node` | `/charging_alignment_server` |
| `/arc/nav_fine_alignment_state` | `robots_dog_msgs/msg/ArcModuleState` | `/charging_alignment_server` | `/arc_state_machine_node` |
| `/arc/perception_dock_pose` | `robots_dog_msgs/msg/DockPoseStamped` | `/aritag_location_pipline` | `/arc_state_machine_node` |
| `/arc/perception_mode_cmd` | `robots_dog_msgs/msg/ArcModuleCmd` | `/arc_state_machine_node` | `/aritag_location_pipline` |
| `/arc/perception_state` | `robots_dog_msgs/msg/ArcModuleState` | `/aritag_location_pipline` | `/arc_state_machine_node` |
| `/arc/pile_ele_cmd` | `std_msgs/msg/Int8` | `/kanon_rust_charge` | — |
| `/arc/pile_ele_result` | `std_msgs/msg/Int8` | — | `/kanon_rust_charge` |
| `/arc/slam_dock_pose` | `robots_dog_msgs/msg/DockPoseStamped` | `/arc_lvio_node` | `/arc_state_machine_node` |
| `/arc/slam_mode_cmd` | `robots_dog_msgs/msg/ArcModuleCmd` | `/arc_state_machine_node` | `/arc_lvio_node` |
| `/arc/slam_state` | `robots_dog_msgs/msg/ArcModuleState` | `/arc_lvio_node` | `/arc_state_machine_node` |
| `/arc/start_arc` | `robots_dog_msgs/msg/StartArc` | `/kanon_rust_charge` | `/arc_state_machine_node` |
| `/arc_mapping_state` | `robots_dog_msgs/msg/SlamState` | `/arc_mapping_node` | `/kanon_rust_charge` |
| `/body_points` | `sensor_msgs/msg/PointCloud2` | `/localization` | — |
| `/cmd_pos` | `geometry_msgs/msg/Pose` | `/charging_alignment_server` | `/remoix_rust_interface` |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `/MEB`、`/charging_alignment_server`、`/velocity_optimizer` | `/remoix_rust_interface` |
| `/cmd_vel_with_mc_trajectory` | `robots_dog_msgs/msg/CmdVelWithTrajectory` | `/controller_server` | — |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | `/lifecycle_manager_navigation` | — |
| `/diagnostics/nav_error_report` | `robots_dog_msgs/msg/NavigationErrorReport` | — | `/rust_debug_node` |
| `/electronic_fence` | `robots_dog_msgs/msg/ElectronicMap` | — | `/global_costmap/global_costmap` |
| `/electronic_map` | `robots_dog_msgs/msg/ElectronicMap` | — | `/local_costmap/local_costmap` |
| `/front_camera/image_compressed` | `sensor_msgs/msg/CompressedImage` | `/bridge_image_topics_24` | `/arc_lvio_node`、`/arc_mapping_node`、`/aritag_location_pipline`、`/localization`、`/perception_jobs`、`/robot_slam` |
| `/front_lidar` | `sensor_msgs/msg/PointCloud2` | `/rslidar_sdk/rslidar_points_destination_0` | `/arc_lvio_node`、`/localization`、`/perception_jobs`、`/robot_slam` |
| `/front_lidar/imu` | `sensor_msgs/msg/Imu` | `/rslidar_sdk/rslidar_points_destination_0` | `/arc_lvio_node`、`/localization`、`/robot_slam` |
| `/global_map_points` | `sensor_msgs/msg/PointCloud2` | `/localization` | — |
| `/gnss/data` | `robots_dog_msgs/msg/Nmea` | — | `/localization` |
| `/goal` | `geometry_msgs/msg/PoseStamped` | — | `/ros2_alg_interface` |
| `/gps/rtk` | `sensor_msgs/msg/NavSatFix` | — | `/localization` |
| `/handle_vel` | `geometry_msgs/msg/Twist` | `/remoix_rust_interface` | `/MEB` |
| `/imu_driver/imu_central` | `sensor_msgs/msg/Imu` | `/imu_driver` | — |
| `/initialpose` | `geometry_msgs/msg/PoseWithCovarianceStamped` | — | `/localization` |
| `/laser_scan` | `sensor_msgs/msg/LaserScan` | `/perception_jobs` | `/MEB`、`/collision_monitor`、`/controller_server`、`/error_aggregator`、`/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller` |
| `/lio_pose` | `nav_msgs/msg/Odometry` | `/localization` | — |
| `/localization_info` | `robots_dog_msgs/msg/Localization` | `/localization` | `/kanon_rust_nav` |
| `/localization_path` | `nav_msgs/msg/Path` | `/localization` | — |
| `/localization_state` | `robots_dog_msgs/msg/LocalizationState` | `/localization` | `/kanon_rust_loc`、`/waypoint_follower` |
| `/meb/collision_debug` | `visualization_msgs/msg/Marker` | `/MEB` | — |
| `/meb/pred_path` | `nav_msgs/msg/Path` | `/MEB` | — |
| `/meb/safety_corridor` | `visualization_msgs/msg/MarkerArray` | `/MEB` | — |
| `/meb_brake` | `std_msgs/msg/Bool` | `/MEB` | `/remoix_rust_interface` |
| `/meb_status` | `std_msgs/msg/UInt8` | `/MEB` | `/ros2_alg_interface` |
| `/meb_switch` | `std_msgs/msg/Bool` | `/ros2_alg_interface` | `/MEB` |
| `/navigation_cmd` | `robots_dog_msgs/msg/NavigationCmd` | `/arc_state_machine_node`、`/charging_alignment_server`、`/velocity_optimizer` | `/remoix_rust_interface` |
| `/navigation_state` | `robots_dog_msgs/msg/NavigationState` | `/waypoint_follower` | `/arc_state_machine_node`、`/kanon_rust_nav` |
| `/navigo/bn/cmn/dbg/behavior_tree_log` | `nav2_msgs/msg/BehaviorTreeLog` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/candidate_local_paths` | `robots_dog_msgs/msg/TrajectoryArray` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/evaluated_goal` | `geometry_msgs/msg/PoseStamped` | `/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/global_path` | `nav_msgs/msg/Path` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | `/ros2_alg_interface` |
| `/navigo/bn/cmn/vis/goal` | `geometry_msgs/msg/PoseStamped` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/goals` | `geometry_msgs/msg/PoseArray` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/local_path` | `nav_msgs/msg/Path` | `/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node` | `/ros2_alg_interface` |
| `/navigo/bn/cmn/vis/predicted_goal` | `geometry_msgs/msg/PoseStamped` | `/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/predicted_goal_path` | `robots_dog_msgs/msg/Trajectory` | `/bt_navigator_navigate_to_pose_rclcpp_node` | — |
| `/navigo/bn/cmn/vis/relocalization_goal` | `visualization_msgs/msg/MarkerArray` | `/bt_navigator_navigate_through_poses_rclcpp_node` | — |
| `/navigo/cm/cmn/vis/polygon_stop` | `geometry_msgs/msg/PolygonStamped` | `/collision_monitor` | — |
| `/navigo/cm/cmn/vis/selective_stop` | `visualization_msgs/msg/MarkerArray` | `/collision_monitor` | — |
| `/navigo/cs/cmn/intf/cmd_vel_raw` | `geometry_msgs/msg/Twist` | `/behavior_server`、`/controller_server` | `/collision_monitor`、`/controller_server` |
| `/navigo/cs/cmn/intf/cmd_vel_valid` | `geometry_msgs/msg/Twist` | `/collision_monitor` | `/velocity_optimizer` |
| `/navigo/cs/lpc/vis/base_candidate_plan` | `nav_msgs/msg/Path` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/best_local_plan` | `nav_msgs/msg/Path` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/footprint_collision` | `visualization_msgs/msg/Marker` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/footprint_samples` | `visualization_msgs/msg/Marker` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/free_paths` | `sensor_msgs/msg/PointCloud2` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/laser_point_cloud` | `sensor_msgs/msg/PointCloud2` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/local_planner_goal` | `geometry_msgs/msg/PoseStamped` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/lookahead_point` | `visualization_msgs/msg/Marker` | `/controller_server` | — |
| `/navigo/cs/lpc/vis/received_global_plan` | `nav_msgs/msg/Path` | `/controller_server` | — |
| `/navigo/cs/ppc/vis/curvature_lookahead_point` | `geometry_msgs/msg/PointStamped` | `/controller_server` | — |
| `/navigo/cs/ppc/vis/lookahead_collision_arc` | `nav_msgs/msg/Path` | `/controller_server` | — |
| `/navigo/cs/ppc/vis/lookahead_point` | `geometry_msgs/msg/PointStamped` | `/controller_server` | — |
| `/navigo/cs/ppc/vis/predicted_points` | `visualization_msgs/msg/MarkerArray` | `/controller_server` | — |
| `/navigo/cs/ppc/vis/received_global_plan` | `nav_msgs/msg/Path` | `/controller_server` | — |
| `/navigo/ea/cmn/intf/nav_error` | `robots_dog_msgs/msg/NavigationError` | `/bt_navigator`、`/collision_monitor`、`/controller_server`、`/planner_server`、`/waypoint_follower` | `/error_aggregator` |
| `/navigo/ea/cmn/intf/nav_error_clear` | `robots_dog_msgs/msg/NavigationError` | `/bt_navigator`、`/collision_monitor`、`/controller_server`、`/planner_server`、`/waypoint_follower` | `/error_aggregator` |
| `/navigo/ea/cmn/intf/nav_error_primary` | `robots_dog_msgs/msg/NavigationError` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_error_report` | `robots_dog_msgs/msg/NavigationErrorReport` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_errors` | `robots_dog_msgs/msg/NavigationErrorSet` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/behavior_server/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/bt_navigator/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/collision_monitor/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/controller_server/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/global_costmap/global_costmap/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/local_costmap/local_costmap/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/local_costmap_controller/local_costmap_controller/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/map_server/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/planner_server/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/velocity_optimizer/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/ea/cmn/intf/nav_lifecycle/waypoint_follower/state` | `lifecycle_msgs/msg/State` | `/error_aggregator` | — |
| `/navigo/gc/cmn/intf/footprint` | `geometry_msgs/msg/Polygon` | — | `/global_costmap/global_costmap` |
| `/navigo/gc/cmn/vis/costmap` | `nav_msgs/msg/OccupancyGrid` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/cmn/vis/costmap_raw` | `nav2_msgs/msg/Costmap` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/cmn/vis/costmap_updates` | `map_msgs/msg/OccupancyGridUpdate` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/cmn/vis/published_footprint` | `geometry_msgs/msg/PolygonStamped` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/ftr/vis/grouped_frontiers_map` | `nav_msgs/msg/OccupancyGrid` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/ftr/vis/include_frontiers_map` | `nav_msgs/msg/OccupancyGrid` | `/global_costmap/global_costmap` | — |
| `/navigo/gc/ftr/vis/processed_map` | `nav_msgs/msg/OccupancyGrid` | `/global_costmap/global_costmap` | — |
| `/navigo/lc/cmn/intf/footprint` | `geometry_msgs/msg/Polygon` | — | `/local_costmap/local_costmap` |
| `/navigo/lc/cmn/vis/costmap` | `nav_msgs/msg/OccupancyGrid` | `/local_costmap/local_costmap` | — |
| `/navigo/lc/cmn/vis/costmap_raw` | `nav2_msgs/msg/Costmap` | `/local_costmap/local_costmap` | — |
| `/navigo/lc/cmn/vis/costmap_updates` | `map_msgs/msg/OccupancyGridUpdate` | `/local_costmap/local_costmap` | — |
| `/navigo/lc/cmn/vis/published_footprint` | `geometry_msgs/msg/PolygonStamped` | `/local_costmap/local_costmap` | — |
| `/navigo/lc/obs/vis/published_goal_safe_area` | `geometry_msgs/msg/PolygonStamped` | `/local_costmap/local_costmap` | — |
| `/navigo/lcc/cmn/intf/footprint` | `geometry_msgs/msg/Polygon` | — | `/local_costmap_controller/local_costmap_controller` |
| `/navigo/lcc/cmn/vis/costmap` | `nav_msgs/msg/OccupancyGrid` | `/local_costmap_controller/local_costmap_controller` | — |
| `/navigo/lcc/cmn/vis/costmap_raw` | `nav2_msgs/msg/Costmap` | `/local_costmap_controller/local_costmap_controller` | `/behavior_server` |
| `/navigo/lcc/cmn/vis/costmap_updates` | `map_msgs/msg/OccupancyGridUpdate` | `/local_costmap_controller/local_costmap_controller` | — |
| `/navigo/lcc/cmn/vis/published_footprint` | `geometry_msgs/msg/PolygonStamped` | `/local_costmap_controller/local_costmap_controller` | `/behavior_server` |
| `/navigo/lcc/obs/vis/published_goal_safe_area` | `geometry_msgs/msg/PolygonStamped` | `/local_costmap_controller/local_costmap_controller` | — |
| `/navigo/lm/cmn/dbg/behavior_server/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/behavior_server` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/bt_navigator/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/bt_navigator` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/collision_monitor/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/collision_monitor` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/controller_server/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/controller_server` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/global_costmap/global_costmap/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/global_costmap/global_costmap` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/local_costmap/local_costmap/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/local_costmap/local_costmap` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/local_costmap_controller/local_costmap_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/local_costmap_controller/local_costmap_controller` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/map_server/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/map_server` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/planner_server/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/planner_server` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/velocity_optimizer/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/velocity_optimizer` | `/error_aggregator` |
| `/navigo/lm/cmn/dbg/waypoint_follower/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/waypoint_follower` | `/error_aggregator` |
| `/navigo/lm/cmn/intf/bond` | `bond/msg/Status` | `/behavior_server`、`/bt_navigator`、`/collision_monitor`、`/controller_server`、`/lifecycle_manager_navigation`、`/map_server`、`/planner_server`、`/velocity_optimizer`、`/waypoint_follower` | `/behavior_server`、`/bt_navigator`、`/collision_monitor`、`/controller_server`、`/lifecycle_manager_navigation`、`/map_server`、`/planner_server`、`/velocity_optimizer`、`/waypoint_follower` |
| `/navigo/ms/cmn/intf/map` | `nav_msgs/msg/OccupancyGrid` | `/map_server` | `/global_costmap/global_costmap` |
| `/navigo/ps/cmn/intf/clean_obstacles_around_goal` | `geometry_msgs/msg/PoseStamped` | `/planner_server` | `/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller` |
| `/navigo/ps/cmn/vis/get_optimized_path_all_paths_visualization` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/get_optimized_path_corridor_visualization` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/get_optimized_path_matched_points_markers` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/global_esdf_pos_distance` | `nav_msgs/msg/OccupancyGrid` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/local_esdf_pos_distance` | `nav_msgs/msg/OccupancyGrid` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/planned_path` | `nav_msgs/msg/Path` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/planned_trajectories` | `robots_dog_msgs/msg/Trajectory` | `/planner_server` | — |
| `/navigo/ps/cmn/vis/quadtree_topo_graph` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/slp/vis/start_end_visualization` | `geometry_msgs/msg/PoseArray` | `/planner_server` | — |
| `/navigo/ps/stp/dbg/planner_debug_info` | `robots_dog_msgs/msg/PlannerDebugInfo` | `/planner_server` | — |
| `/navigo/ps/stp/vis/corridor_clearance_visualization` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_exploration_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_goal_connection_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_middle_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_shot_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_start_connection_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/front_end_traj_visualization` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/guidance_path` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_exploration_points` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/candidate_inputs` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/cost_indicators` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/direction_arrows` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/explored_nodes` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/robot_footprint` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/selected_input` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_search/shot_path` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/hybrid_astar_start_end_sampling_points` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/quadtree_dijkstra_progress` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/quadtree_multi_inflation_results` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/quadtree_path_visualization` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/quadtree_start_goal_grids` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/safe_drive_corridor` | `robots_dog_msgs/msg/RectangleArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/start_end_visualization` | `geometry_msgs/msg/PoseArray` | `/planner_server` | — |
| `/navigo/ps/stp/vis/tracking_searcher_each_expansion_step_planner_goal` | `geometry_msgs/msg/PoseStamped` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/centerline_points` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/esdf_cloud` | `sensor_msgs/msg/PointCloud2` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/esdf_slice` | `nav_msgs/msg/OccupancyGrid` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/pcd_overlay` | `sensor_msgs/msg/PointCloud2` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/three_dim_search_path` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/three_dim_topo_edges` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/three_dim_topo_nodes` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/ps/tpg/vis/virtual_walls` | `visualization_msgs/msg/MarkerArray` | `/planner_server` | — |
| `/navigo/wf/cmn/intf/update_tracking_goal` | `geometry_msgs/msg/PoseStamped` | `/waypoint_follower` | `/bt_navigator` |
| `/navigo/wf/cmn/vis/original_reference_path` | `nav_msgs/msg/Path` | `/waypoint_follower` | — |
| `/odom/current_pose` | `nav_msgs/msg/Odometry` | `/robot_tf` | `/arc_state_machine_node`、`/behavior_server`、`/bt_navigator`、`/collision_monitor`、`/controller_server`、`/error_aggregator`、`/global_costmap/global_costmap`、`/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller`、`/planner_server`、`/ros2_alg_interface`、`/waypoint_follower` |
| `/odom/current_pose/history` | `nav_msgs/msg/Path` | `/waypoint_follower` | — |
| `/odom/gazebo_odom` | `nav_msgs/msg/Odometry` | — | `/robot_tf` |
| `/odom/localization_odom` | `nav_msgs/msg/Odometry` | `/localization` | `/arc_mapping_node`、`/robot_tf` |
| `/odom/mc_odom` | `nav_msgs/msg/Odometry` | `/remoix_rust_interface` | `/localization`、`/robot_tf` |
| `/odom/mujoco_odom` | `nav_msgs/msg/Odometry` | — | `/robot_tf` |
| `/odom/slam_odom` | `nav_msgs/msg/Odometry` | `/robot_slam` | `/robot_tf` |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | `/MEB`、`/arc_lvio_node`、`/arc_mapping_node`、`/arc_state_machine_node`、`/aritag_location_pipline`、`/behavior_server`、`/bt_navigator`、`/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node`、`/charging_alignment_server`、`/collision_monitor`、`/controller_server`、`/error_aggregator`、`/global_costmap/global_costmap`、`/imu_driver`、`/launch_ros_2483`、`/lifecycle_manager_navigation`、`/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller`、`/localization`、`/map_server`、`/perception_jobs`、`/planner_server`、`/robot_slam`、`/robot_tf`、`/rslidar_sdk/param_handle`、`/rslidar_sdk/rslidar_points_destination_0`、`/rslidar_sdk/rslidar_points_destination_1`、`/sixents_gps_driver`、`/uss_driver`、`/uwb_driver`、`/velocity_optimizer`、`/waypoint_follower` | `/MEB`、`/arc_lvio_node`、`/arc_mapping_node`、`/arc_state_machine_node`、`/aritag_location_pipline`、`/behavior_server`、`/bridge_image_topics_24`、`/bt_navigator`、`/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node`、`/charging_alignment_server`、`/collision_monitor`、`/controller_server`、`/error_aggregator`、`/global_costmap/global_costmap`、`/imu_driver`、`/lifecycle_manager_navigation`、`/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller`、`/localization`、`/map_server`、`/nav2_container`、`/perception_jobs`、`/planner_server`、`/robot_slam`、`/robot_tf`、`/rslidar_sdk/param_handle`、`/rslidar_sdk/rslidar_points_destination_0`、`/rslidar_sdk/rslidar_points_destination_1`、`/sixents_gps_driver`、`/uss_driver`、`/uwb_driver`、`/velocity_optimizer`、`/waypoint_follower` |
| `/perception/detection3d` | `vision_msgs/msg/Detection3DArray` | — | `/waypoint_follower` |
| `/perception_points` | `robots_dog_msgs/msg/PerceptionPoints` | `/localization`、`/robot_slam` | — |
| `/perception_state` | `std_msgs/msg/Int32` | `/perception_jobs` | — |
| `/pers/state` | `robots_dog_msgs/msg/PerceptionState` | — | `/waypoint_follower` |
| `/polygons` | `robots_dog_msgs/msg/RectangleArray` | `/perception_jobs` | — |
| `/predicted_paths` | `robots_dog_msgs/msg/PredictedPathArray` | — | `/planner_server` |
| `/pub_slam_state` | `robots_dog_msgs/msg/SlamState` | `/robot_slam` | `/kanon_rust_slam` |
| `/rear_camera/image_compressed` | `sensor_msgs/msg/CompressedImage` | `/bridge_image_topics_24` | — |
| `/rear_lidar` | `sensor_msgs/msg/PointCloud2` | `/rslidar_sdk/rslidar_points_destination_1` | `/arc_lvio_node`、`/localization`、`/robot_slam` |
| `/rear_lidar/imu` | `sensor_msgs/msg/Imu` | `/rslidar_sdk/rslidar_points_destination_1` | — |
| `/rosout` | `rcl_interfaces/msg/Log` | `/MEB`、`/arc_lvio_node`、`/arc_mapping_node`、`/arc_state_machine_node`、`/aritag_location_pipline`、`/behavior_server`、`/bridge_image_topics_24`、`/bt_navigator`、`/bt_navigator_navigate_through_poses_rclcpp_node`、`/bt_navigator_navigate_to_pose_rclcpp_node`、`/charging_alignment_server`、`/collision_monitor`、`/controller_server`、`/error_aggregator`、`/global_costmap/global_costmap`、`/imu_driver`、`/kanon_rust_charge`、`/kanon_rust_loc`、`/kanon_rust_nav`、`/kanon_rust_slam`、`/launch_ros_2483`、`/lifecycle_manager_navigation`、`/local_costmap/local_costmap`、`/local_costmap_controller/local_costmap_controller`、`/localization`、`/map_server`、`/nav2_container`、`/perception_jobs`、`/planner_server`、`/remoix_rust_interface`、`/robot_slam`、`/robot_tf`、`/ros2_alg_interface`、`/rslidar_sdk/param_handle`、`/rslidar_sdk/rslidar_points_destination_0`、`/rslidar_sdk/rslidar_points_destination_1`、`/rust_debug_node`、`/sixents_gps_driver`、`/uss_driver`、`/uwb_driver`、`/velocity_optimizer`、`/waypoint_follower` | — |
| `/rtk_pvh` | `robots_dog_msgs/msg/UniRtkPvh` | `/sixents_gps_driver` | — |
| `/seg_vis/compressed` | `sensor_msgs/msg/CompressedImage` | `/perception_jobs` | — |
| `/set_desired_angular_vel_z` | `std_msgs/msg/Float32` | — | `/bt_navigator` |
| `/set_desired_linear_vel_x` | `std_msgs/msg/Float32` | — | `/bt_navigator` |
| `/set_desired_linear_vel_y` | `std_msgs/msg/Float32` | — | `/bt_navigator` |
| `/set_lookahead_distance` | `std_msgs/msg/Float32` | — | `/controller_server` |
| `/set_speed_limit` | `nav2_msgs/msg/SpeedLimit` | — | `/controller_server` |
| `/slam/manual_loop` | `std_msgs/msg/Float64MultiArray` | — | `/robot_slam` |
| `/srv/event` | `robots_dog_msgs/msg/SrvEvent` | `/ros2_alg_interface` | — |
| `/start_navigation` | `robots_dog_msgs/msg/StartNavigation` | `/arc_state_machine_node`、`/kanon_rust_nav` | `/waypoint_follower` |
| `/tf` | `tf2_msgs/msg/TFMessage` | `/arc_lvio_node`、`/arc_mapping_node`、`/aritag_location_pipline`、`/localization`、`/robot_tf` | — |
| `/tf_manager/odom_type` | `std_msgs/msg/Header` | `/robot_tf` | — |
| `/tf_static` | `tf2_msgs/msg/TFMessage` | `/aritag_location_pipline`、`/robot_tf` | — |
| `/uni_best_nav` | `robots_dog_msgs/msg/UniBestNav` | `/sixents_gps_driver` | `/localization` |
| `/uni_heading` | `robots_dog_msgs/msg/UniHeading` | `/sixents_gps_driver` | `/localization` |
| `/uni_rtk_pvh` | `robots_dog_msgs/msg/UniRtkPvh` | — | `/localization` |
| `/uss_driver/uss_left/range` | `sensor_msgs/msg/Range` | `/uss_driver` | — |
| `/uss_driver/uss_right/range` | `sensor_msgs/msg/Range` | `/uss_driver` | — |
| `/uwb` | `uwb_driver/msg/Uwb` | `/uwb_driver` | — |
| `/uwb_point` | `geometry_msgs/msg/PointStamped` | — | `/waypoint_follower` |
| `/world_points` | `sensor_msgs/msg/PointCloud2` | `/robot_slam` | — |

### 每個節點的完整接線

**`/MEB`**（導航）
- 發布：`/cmd_vel`、`/meb/collision_debug`、`/meb/pred_path`、`/meb/safety_corridor`、`/meb_brake`、`/meb_status`、`/parameter_events`、`/rosout`
- 訂閱：`/handle_vel`、`/laser_scan`、`/meb_switch`、`/parameter_events`

**`/arc_lvio_node`**（SLAM／定位）
- 發布：`/arc/slam_dock_pose`、`/arc/slam_state`、`/parameter_events`、`/rosout`、`/tf`
- 訂閱：`/arc/slam_mode_cmd`、`/front_camera/image_compressed`、`/front_lidar`、`/front_lidar/imu`、`/parameter_events`、`/rear_lidar`

**`/arc_mapping_node`**（SLAM／定位）
- 發布：`/arc_mapping_state`、`/parameter_events`、`/rosout`、`/tf`
- 訂閱：`/front_camera/image_compressed`、`/odom/localization_odom`、`/parameter_events`

**`/arc_state_machine_node`**（其他）
- 發布：`/arc/arc_change_flag`、`/arc/arc_module_states_debug_info`、`/arc/arc_state`、`/arc/arc_state_debug_info`、`/arc/dock_pose`、`/arc/mc_mode_cmd`、`/arc/nav_coarse_alignment_mode_cmd`、`/arc/nav_contact_alignment_mode_cmd`、`/arc/nav_fine_alignment_mode_cmd`、`/arc/perception_mode_cmd`、`/arc/slam_mode_cmd`、`/navigation_cmd`、`/parameter_events`、`/rosout`、`/start_navigation`
- 訂閱：`/arc/arc_change_flag_test`、`/arc/calibration_state`、`/arc/dock_state`、`/arc/mc_state`、`/arc/nav_coarse_alignment_state`、`/arc/nav_contact_alignment_state`、`/arc/nav_fine_alignment_state`、`/arc/perception_dock_pose`、`/arc/perception_state`、`/arc/slam_dock_pose`、`/arc/slam_state`、`/arc/start_arc`、`/navigation_state`、`/odom/current_pose`、`/parameter_events`

**`/aritag_location_pipline`**（SLAM／定位）
- 發布：`/arc/calibration_state`、`/arc/perception_dock_pose`、`/arc/perception_state`、`/parameter_events`、`/rosout`、`/tf`、`/tf_static`
- 訂閱：`/arc/arc_state`、`/arc/perception_mode_cmd`、`/front_camera/image_compressed`、`/parameter_events`

**`/behavior_server`**（導航）
- 發布：`/navigo/cs/cmn/intf/cmd_vel_raw`、`/navigo/lm/cmn/dbg/behavior_server/transition_event`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/lcc/cmn/vis/costmap_raw`、`/navigo/lcc/cmn/vis/published_footprint`、`/navigo/lm/cmn/intf/bond`、`/odom/current_pose`、`/parameter_events`

**`/bridge_image_topics_24`**（其他）
- 發布：`/front_camera/image_compressed`、`/rear_camera/image_compressed`、`/rosout`
- 訂閱：`/parameter_events`

**`/bt_navigator`**（導航）
- 發布：`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/bt_navigator/transition_event`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/lm/cmn/intf/bond`、`/navigo/wf/cmn/intf/update_tracking_goal`、`/odom/current_pose`、`/parameter_events`、`/set_desired_angular_vel_z`、`/set_desired_linear_vel_x`、`/set_desired_linear_vel_y`

**`/bt_navigator_navigate_through_poses_rclcpp_node`**（導航）
- 發布：`/navigo/bn/cmn/dbg/behavior_tree_log`、`/navigo/bn/cmn/vis/candidate_local_paths`、`/navigo/bn/cmn/vis/global_path`、`/navigo/bn/cmn/vis/goal`、`/navigo/bn/cmn/vis/goals`、`/navigo/bn/cmn/vis/local_path`、`/navigo/bn/cmn/vis/relocalization_goal`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/bt_navigator_navigate_to_pose_rclcpp_node`**（導航）
- 發布：`/navigo/bn/cmn/dbg/behavior_tree_log`、`/navigo/bn/cmn/vis/candidate_local_paths`、`/navigo/bn/cmn/vis/evaluated_goal`、`/navigo/bn/cmn/vis/global_path`、`/navigo/bn/cmn/vis/goal`、`/navigo/bn/cmn/vis/goals`、`/navigo/bn/cmn/vis/local_path`、`/navigo/bn/cmn/vis/predicted_goal`、`/navigo/bn/cmn/vis/predicted_goal_path`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/charging_alignment_server`**（導航）
- 發布：`/arc/nav_alignment/cmd_vel_raw`、`/arc/nav_alignment/debug_info`、`/arc/nav_alignment/error_vector`、`/arc/nav_coarse_alignment_state`、`/arc/nav_contact_alignment_state`、`/arc/nav_fine_alignment_state`、`/cmd_pos`、`/cmd_vel`、`/navigation_cmd`、`/parameter_events`、`/rosout`
- 訂閱：`/arc/dock_pose`、`/arc/nav_coarse_alignment_mode_cmd`、`/arc/nav_contact_alignment_mode_cmd`、`/arc/nav_fine_alignment_mode_cmd`、`/parameter_events`

**`/collision_monitor`**（導航）
- 發布：`/navigo/cm/cmn/vis/polygon_stop`、`/navigo/cm/cmn/vis/selective_stop`、`/navigo/cs/cmn/intf/cmd_vel_valid`、`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/collision_monitor/transition_event`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/laser_scan`、`/navigo/cs/cmn/intf/cmd_vel_raw`、`/navigo/lm/cmn/intf/bond`、`/odom/current_pose`、`/parameter_events`

**`/controller_server`**（導航）
- 發布：`/cmd_vel_with_mc_trajectory`、`/navigo/cs/cmn/intf/cmd_vel_raw`、`/navigo/cs/lpc/vis/base_candidate_plan`、`/navigo/cs/lpc/vis/best_local_plan`、`/navigo/cs/lpc/vis/footprint_collision`、`/navigo/cs/lpc/vis/footprint_samples`、`/navigo/cs/lpc/vis/free_paths`、`/navigo/cs/lpc/vis/laser_point_cloud`、`/navigo/cs/lpc/vis/local_planner_goal`、`/navigo/cs/lpc/vis/lookahead_point`、`/navigo/cs/lpc/vis/received_global_plan`、`/navigo/cs/ppc/vis/curvature_lookahead_point`、`/navigo/cs/ppc/vis/lookahead_collision_arc`、`/navigo/cs/ppc/vis/lookahead_point`、`/navigo/cs/ppc/vis/predicted_points`、`/navigo/cs/ppc/vis/received_global_plan`、`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/controller_server/transition_event`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/laser_scan`、`/navigo/cs/cmn/intf/cmd_vel_raw`、`/navigo/lm/cmn/intf/bond`、`/odom/current_pose`、`/parameter_events`、`/set_lookahead_distance`、`/set_speed_limit`

**`/error_aggregator`**（其他）
- 發布：`/navigo/ea/cmn/intf/nav_error_primary`、`/navigo/ea/cmn/intf/nav_error_report`、`/navigo/ea/cmn/intf/nav_errors`、`/navigo/ea/cmn/intf/nav_lifecycle/behavior_server/state`、`/navigo/ea/cmn/intf/nav_lifecycle/bt_navigator/state`、`/navigo/ea/cmn/intf/nav_lifecycle/collision_monitor/state`、`/navigo/ea/cmn/intf/nav_lifecycle/controller_server/state`、`/navigo/ea/cmn/intf/nav_lifecycle/global_costmap/global_costmap/state`、`/navigo/ea/cmn/intf/nav_lifecycle/local_costmap/local_costmap/state`、`/navigo/ea/cmn/intf/nav_lifecycle/local_costmap_controller/local_costmap_controller/state`、`/navigo/ea/cmn/intf/nav_lifecycle/map_server/state`、`/navigo/ea/cmn/intf/nav_lifecycle/planner_server/state`、`/navigo/ea/cmn/intf/nav_lifecycle/velocity_optimizer/state`、`/navigo/ea/cmn/intf/nav_lifecycle/waypoint_follower/state`、`/parameter_events`、`/rosout`
- 訂閱：`/laser_scan`、`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/behavior_server/transition_event`、`/navigo/lm/cmn/dbg/bt_navigator/transition_event`、`/navigo/lm/cmn/dbg/collision_monitor/transition_event`、`/navigo/lm/cmn/dbg/controller_server/transition_event`、`/navigo/lm/cmn/dbg/global_costmap/global_costmap/transition_event`、`/navigo/lm/cmn/dbg/local_costmap/local_costmap/transition_event`、`/navigo/lm/cmn/dbg/local_costmap_controller/local_costmap_controller/transition_event`、`/navigo/lm/cmn/dbg/map_server/transition_event`、`/navigo/lm/cmn/dbg/planner_server/transition_event`、`/navigo/lm/cmn/dbg/velocity_optimizer/transition_event`、`/navigo/lm/cmn/dbg/waypoint_follower/transition_event`、`/odom/current_pose`、`/parameter_events`

**`/global_costmap/global_costmap`**（導航）
- 發布：`/navigo/gc/cmn/vis/costmap`、`/navigo/gc/cmn/vis/costmap_raw`、`/navigo/gc/cmn/vis/costmap_updates`、`/navigo/gc/cmn/vis/published_footprint`、`/navigo/gc/ftr/vis/grouped_frontiers_map`、`/navigo/gc/ftr/vis/include_frontiers_map`、`/navigo/gc/ftr/vis/processed_map`、`/navigo/lm/cmn/dbg/global_costmap/global_costmap/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/electronic_fence`、`/navigo/gc/cmn/intf/footprint`、`/navigo/ms/cmn/intf/map`、`/odom/current_pose`、`/parameter_events`

**`/imu_driver`**（感測驅動）
- 發布：`/imu_driver/imu_central`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/kanon_rust_charge`**（導航）
- 發布：`/arc/pile_ele_cmd`、`/arc/start_arc`、`/rosout`
- 訂閱：`/arc/arc_state`、`/arc/dock_state`、`/arc/pile_ele_result`、`/arc_mapping_state`

**`/kanon_rust_loc`**（導航）
- 發布：`/rosout`
- 訂閱：`/localization_state`

**`/kanon_rust_nav`**（導航）
- 發布：`/rosout`、`/start_navigation`
- 訂閱：`/localization_info`、`/navigation_state`

**`/kanon_rust_slam`**（SLAM／定位）
- 發布：`/rosout`
- 訂閱：`/pub_slam_state`

**`/launch_ros_2483`**（其他）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：—

**`/lifecycle_manager_navigation`**（導航）
- 發布：`/diagnostics`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/lm/cmn/intf/bond`、`/parameter_events`

**`/local_costmap/local_costmap`**（導航）
- 發布：`/navigo/lc/cmn/vis/costmap`、`/navigo/lc/cmn/vis/costmap_raw`、`/navigo/lc/cmn/vis/costmap_updates`、`/navigo/lc/cmn/vis/published_footprint`、`/navigo/lc/obs/vis/published_goal_safe_area`、`/navigo/lm/cmn/dbg/local_costmap/local_costmap/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/electronic_map`、`/laser_scan`、`/navigo/lc/cmn/intf/footprint`、`/navigo/ps/cmn/intf/clean_obstacles_around_goal`、`/odom/current_pose`、`/parameter_events`

**`/local_costmap_controller/local_costmap_controller`**（導航）
- 發布：`/navigo/lcc/cmn/vis/costmap`、`/navigo/lcc/cmn/vis/costmap_raw`、`/navigo/lcc/cmn/vis/costmap_updates`、`/navigo/lcc/cmn/vis/published_footprint`、`/navigo/lcc/obs/vis/published_goal_safe_area`、`/navigo/lm/cmn/dbg/local_costmap_controller/local_costmap_controller/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/laser_scan`、`/navigo/lcc/cmn/intf/footprint`、`/navigo/ps/cmn/intf/clean_obstacles_around_goal`、`/odom/current_pose`、`/parameter_events`

**`/localization`**（SLAM／定位）
- 發布：`/aligned_points`、`/body_points`、`/global_map_points`、`/lio_pose`、`/localization_info`、`/localization_path`、`/localization_state`、`/odom/localization_odom`、`/parameter_events`、`/perception_points`、`/rosout`、`/tf`
- 訂閱：`/front_camera/image_compressed`、`/front_lidar`、`/front_lidar/imu`、`/gnss/data`、`/gps/rtk`、`/initialpose`、`/odom/mc_odom`、`/parameter_events`、`/rear_lidar`、`/uni_best_nav`、`/uni_heading`、`/uni_rtk_pvh`

**`/map_server`**（導航）
- 發布：`/navigo/lm/cmn/dbg/map_server/transition_event`、`/navigo/lm/cmn/intf/bond`、`/navigo/ms/cmn/intf/map`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/lm/cmn/intf/bond`、`/parameter_events`

**`/nav2_container`**（導航）
- 發布：`/rosout`
- 訂閱：`/parameter_events`

**`/perception_jobs`**（感知）
- 發布：`/laser_scan`、`/parameter_events`、`/perception_state`、`/polygons`、`/rosout`、`/seg_vis/compressed`
- 訂閱：`/front_camera/image_compressed`、`/front_lidar`、`/parameter_events`

**`/planner_server`**（導航）
- 發布：`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/planner_server/transition_event`、`/navigo/lm/cmn/intf/bond`、`/navigo/ps/cmn/intf/clean_obstacles_around_goal`、`/navigo/ps/cmn/vis/get_optimized_path_all_paths_visualization`、`/navigo/ps/cmn/vis/get_optimized_path_corridor_visualization`、`/navigo/ps/cmn/vis/get_optimized_path_matched_points_markers`、`/navigo/ps/cmn/vis/global_esdf_pos_distance`、`/navigo/ps/cmn/vis/local_esdf_pos_distance`、`/navigo/ps/cmn/vis/planned_path`、`/navigo/ps/cmn/vis/planned_trajectories`、`/navigo/ps/cmn/vis/quadtree_topo_graph`、`/navigo/ps/slp/vis/start_end_visualization`、`/navigo/ps/stp/dbg/planner_debug_info`、`/navigo/ps/stp/vis/corridor_clearance_visualization`、`/navigo/ps/stp/vis/front_end_exploration_traj_visualization`、`/navigo/ps/stp/vis/front_end_goal_connection_traj_visualization`、`/navigo/ps/stp/vis/front_end_middle_traj_visualization`、`/navigo/ps/stp/vis/front_end_shot_traj_visualization`、`/navigo/ps/stp/vis/front_end_start_connection_traj_visualization`、`/navigo/ps/stp/vis/front_end_traj_visualization`、`/navigo/ps/stp/vis/guidance_path`、`/navigo/ps/stp/vis/hybrid_astar_exploration_points`、`/navigo/ps/stp/vis/hybrid_astar_search/candidate_inputs`、`/navigo/ps/stp/vis/hybrid_astar_search/cost_indicators`、`/navigo/ps/stp/vis/hybrid_astar_search/direction_arrows`、`/navigo/ps/stp/vis/hybrid_astar_search/explored_nodes`、`/navigo/ps/stp/vis/hybrid_astar_search/robot_footprint`、`/navigo/ps/stp/vis/hybrid_astar_search/selected_input`、`/navigo/ps/stp/vis/hybrid_astar_search/shot_path`、`/navigo/ps/stp/vis/hybrid_astar_start_end_sampling_points`、`/navigo/ps/stp/vis/quadtree_dijkstra_progress`、`/navigo/ps/stp/vis/quadtree_multi_inflation_results`、`/navigo/ps/stp/vis/quadtree_path_visualization`、`/navigo/ps/stp/vis/quadtree_start_goal_grids`、`/navigo/ps/stp/vis/safe_drive_corridor`、`/navigo/ps/stp/vis/start_end_visualization`、`/navigo/ps/stp/vis/tracking_searcher_each_expansion_step_planner_goal`、`/navigo/ps/tpg/vis/centerline_points`、`/navigo/ps/tpg/vis/esdf_cloud`、`/navigo/ps/tpg/vis/esdf_slice`、`/navigo/ps/tpg/vis/pcd_overlay`、`/navigo/ps/tpg/vis/three_dim_search_path`、`/navigo/ps/tpg/vis/three_dim_topo_edges`、`/navigo/ps/tpg/vis/three_dim_topo_nodes`、`/navigo/ps/tpg/vis/virtual_walls`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/lm/cmn/intf/bond`、`/odom/current_pose`、`/parameter_events`、`/predicted_paths`

**`/remoix_rust_interface`**（其他）
- 發布：`/arc/dock_state`、`/arc/mc_state`、`/handle_vel`、`/odom/mc_odom`、`/rosout`
- 訂閱：`/arc/arc_state`、`/arc/mc_mode_cmd`、`/cmd_pos`、`/cmd_vel`、`/meb_brake`、`/navigation_cmd`

**`/robot_slam`**（SLAM／定位）
- 發布：`/odom/slam_odom`、`/parameter_events`、`/perception_points`、`/pub_slam_state`、`/rosout`、`/world_points`
- 訂閱：`/front_camera/image_compressed`、`/front_lidar`、`/front_lidar/imu`、`/parameter_events`、`/rear_lidar`、`/slam/manual_loop`

**`/robot_tf`**（SLAM／定位）
- 發布：`/odom/current_pose`、`/parameter_events`、`/rosout`、`/tf`、`/tf_manager/odom_type`、`/tf_static`
- 訂閱：`/odom/gazebo_odom`、`/odom/localization_odom`、`/odom/mc_odom`、`/odom/mujoco_odom`、`/odom/slam_odom`、`/parameter_events`

**`/ros2_alg_interface`**（其他）
- 發布：`/meb_switch`、`/rosout`、`/srv/event`
- 訂閱：`/goal`、`/meb_status`、`/navigo/bn/cmn/vis/global_path`、`/navigo/bn/cmn/vis/local_path`、`/odom/current_pose`

**`/rslidar_sdk/param_handle`**（感測驅動）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/rslidar_sdk/rslidar_points_destination_0`**（感測驅動）
- 發布：`/front_lidar`、`/front_lidar/imu`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/rslidar_sdk/rslidar_points_destination_1`**（感測驅動）
- 發布：`/parameter_events`、`/rear_lidar`、`/rear_lidar/imu`、`/rosout`
- 訂閱：`/parameter_events`

**`/rust_debug_node`**（其他）
- 發布：`/rosout`
- 訂閱：`/arc/arc_module_states_debug_info`、`/diagnostics/nav_error_report`

**`/sixents_gps_driver`**（感測驅動）
- 發布：`/parameter_events`、`/rosout`、`/rtk_pvh`、`/uni_best_nav`、`/uni_heading`
- 訂閱：`/parameter_events`

**`/uss_driver`**（感測驅動）
- 發布：`/parameter_events`、`/rosout`、`/uss_driver/uss_left/range`、`/uss_driver/uss_right/range`
- 訂閱：`/parameter_events`

**`/uwb_driver`**（感測驅動）
- 發布：`/parameter_events`、`/rosout`、`/uwb`
- 訂閱：`/parameter_events`

**`/velocity_optimizer`**（導航）
- 發布：`/cmd_vel`、`/navigation_cmd`、`/navigo/lm/cmn/dbg/velocity_optimizer/transition_event`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`、`/rosout`
- 訂閱：`/navigo/cs/cmn/intf/cmd_vel_valid`、`/navigo/lm/cmn/intf/bond`、`/parameter_events`

**`/waypoint_follower`**（導航）
- 發布：`/navigation_state`、`/navigo/ea/cmn/intf/nav_error`、`/navigo/ea/cmn/intf/nav_error_clear`、`/navigo/lm/cmn/dbg/waypoint_follower/transition_event`、`/navigo/lm/cmn/intf/bond`、`/navigo/wf/cmn/intf/update_tracking_goal`、`/navigo/wf/cmn/vis/original_reference_path`、`/odom/current_pose/history`、`/parameter_events`、`/rosout`
- 訂閱：`/localization_state`、`/navigo/lm/cmn/intf/bond`、`/odom/current_pose`、`/parameter_events`、`/perception/detection3d`、`/pers/state`、`/start_navigation`、`/uwb_point`

---

## RK3588（運控板）：21 個節點、56 個 topic

### 節點 → 發布 / 訂閱

| 節點 | 分類 | 發布 | 訂閱 | 服務 | 動作 |
|---|---|---|---|---|---|
| `/ComponentManager` | 其他 | 1 | 1 | 0 | 0 |
| `/battery_controller` | 導航 | 6 | 1 | 6 | 0 |
| `/bridge_image_topics_66` | 其他 | 1 | 3 | 0 | 0 |
| `/controller_manager` | 導航 | 2 | 1 | 16 | 0 |
| `/fill_light_controller` | 導航 | 7 | 4 | 6 | 0 |
| `/imu_shm_publisher` | 運控／HAL | 4 | 1 | 6 | 0 |
| `/joint_shm_controller` | 導航 | 6 | 2 | 6 | 0 |
| `/launch_ros_2132` | 其他 | 2 | 1 | 7 | 0 |
| `/led_controller` | 導航 | 4 | 15 | 6 | 0 |
| `/robot_camera` | 感測驅動 | 8 | 4 | 7 | 0 |
| `/robot_diagnostic_analyzer` | 其他 | 4 | 2 | 6 | 0 |
| `/robot_diagnostic_manager` | 其他 | 3 | 2 | 6 | 0 |
| `/robot_hal` | 運控／HAL | 4 | 1 | 12 | 0 |
| `/robot_manager` | 運控／HAL | 4 | 3 | 6 | 0 |
| `/robot_monitor` | 運控／HAL | 4 | 1 | 6 | 0 |
| `/robot_remote` | 運控／HAL | 15 | 19 | 13 | 0 |
| `/robot_roamerx` | 其他 | 8 | 6 | 12 | 0 |
| `/robot_self_test_manager` | 其他 | 2 | 1 | 7 | 0 |
| `/switch_controller` | 導航 | 4 | 1 | 6 | 0 |
| `/zsi_actuator_driver` | 其他 | 2 | 1 | 6 | 0 |
| `/zsi_imu_driver` | 感測驅動 | 2 | 1 | 6 | 0 |

### Topic → 發布者 / 訂閱者

| Topic | 型別 | 發布者 | 訂閱者 |
|---|---|---|---|
| `/arc/charging_dock_info` | `robot_common_interface/msg/ChargingDockInfo` | `/robot_roamerx` | `/robot_remote` |
| `/battery_controller/battery1` | `sensor_msgs/msg/BatteryState` | `/battery_controller` | `/led_controller`、`/robot_remote`、`/robot_roamerx` |
| `/battery_controller/battery2` | `sensor_msgs/msg/BatteryState` | `/battery_controller` | `/led_controller`、`/robot_remote`、`/robot_roamerx` |
| `/battery_controller/battery_all` | `sensor_msgs/msg/BatteryState` | `/battery_controller` | `/led_controller` |
| `/battery_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/battery_controller` | — |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | — | `/robot_roamerx` |
| `/control_right/test` | `std_msgs/msg/Bool` | — | `/robot_roamerx` |
| `/diagnostics` | `diagnostic_msgs/msg/DiagnosticArray` | `/robot_hal`、`/robot_manager`、`/robot_monitor`、`/robot_remote`、`/robot_roamerx` | `/robot_diagnostic_analyzer` |
| `/diagnostics_agg` | `diagnostic_msgs/msg/DiagnosticArray` | `/robot_diagnostic_analyzer` | `/robot_diagnostic_manager`、`/robot_remote` |
| `/diagnostics_toplevel_state` | `diagnostic_msgs/msg/DiagnosticStatus` | `/robot_diagnostic_analyzer` | — |
| `/estop_controller/hw_estop/state` | `std_msgs/msg/Bool` | — | `/led_controller` |
| `/estop_controller/sw_estop/state` | `std_msgs/msg/Bool` | `/robot_remote` | `/led_controller` |
| `/fill_light_controller/auto_work` | `std_msgs/msg/Bool` | `/robot_remote` | `/fill_light_controller` |
| `/fill_light_controller/auto_work_state` | `std_msgs/msg/Bool` | `/fill_light_controller` | `/robot_remote` |
| `/fill_light_controller/back_light/cmd` | `std_msgs/msg/Bool` | `/robot_remote` | `/fill_light_controller` |
| `/fill_light_controller/back_light/state` | `std_msgs/msg/Bool` | `/fill_light_controller` | `/robot_remote` |
| `/fill_light_controller/front_light/cmd` | `std_msgs/msg/Bool` | `/robot_remote` | `/fill_light_controller` |
| `/fill_light_controller/front_light/state` | `std_msgs/msg/Bool` | `/fill_light_controller` | `/robot_remote` |
| `/fill_light_controller/lux` | `sensor_msgs/msg/Illuminance` | `/fill_light_controller` | `/robot_remote` |
| `/fill_light_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/fill_light_controller` | — |
| `/front_camera/image_compressed` | `sensor_msgs/msg/CompressedImage` | `/robot_camera` | `/bridge_image_topics_66` |
| `/imu_shm_publisher/imu_central` | `sensor_msgs/msg/Imu` | `/imu_shm_publisher` | `/robot_remote`、`/robot_roamerx` |
| `/imu_shm_publisher/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/imu_shm_publisher` | — |
| `/joint_shm_controller/joint_cmd_echo` | `sensor_msgs/msg/JointState` | `/joint_shm_controller` | — |
| `/joint_shm_controller/joint_sensor` | `robot_common_interface/msg/JointSensor` | `/joint_shm_controller` | `/robot_remote` |
| `/joint_shm_controller/joint_states` | `sensor_msgs/msg/JointState` | `/joint_shm_controller` | — |
| `/joint_shm_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/joint_shm_controller` | — |
| `/led_controller/auto_work` | `std_msgs/msg/Bool` | — | `/led_controller` |
| `/led_controller/auto_work_state` | `std_msgs/msg/Bool` | `/led_controller` | — |
| `/led_controller/led_cmd` | `robot_common_interface/msg/LedCommand` | — | `/led_controller` |
| `/led_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/led_controller` | — |
| `/parameter_events` | `rcl_interfaces/msg/ParameterEvent` | `/battery_controller`、`/controller_manager`、`/fill_light_controller`、`/imu_shm_publisher`、`/joint_shm_controller`、`/launch_ros_2132`、`/led_controller`、`/robot_camera`、`/robot_diagnostic_analyzer`、`/robot_diagnostic_manager`、`/robot_hal`、`/robot_manager`、`/robot_monitor`、`/robot_remote`、`/robot_roamerx`、`/robot_self_test_manager`、`/switch_controller`、`/zsi_actuator_driver`、`/zsi_imu_driver` | `/ComponentManager`、`/battery_controller`、`/bridge_image_topics_66`、`/controller_manager`、`/fill_light_controller`、`/imu_shm_publisher`、`/joint_shm_controller`、`/led_controller`、`/robot_camera`、`/robot_diagnostic_analyzer`、`/robot_diagnostic_manager`、`/robot_hal`、`/robot_manager`、`/robot_monitor`、`/robot_remote`、`/robot_roamerx`、`/robot_self_test_manager`、`/switch_controller`、`/zsi_actuator_driver`、`/zsi_imu_driver` |
| `/rear_camera/image_compressed` | `sensor_msgs/msg/CompressedImage` | `/robot_camera` | `/bridge_image_topics_66` |
| `/robot_camera/record_mp4` | `robot_common_interface/msg/RecordMp4Request` | `/robot_remote` | `/robot_camera` |
| `/robot_camera/record_mp4_ack` | `robot_common_interface/msg/RecordMp4Ack` | `/robot_camera` | `/robot_remote` |
| `/robot_camera/record_mp4_status` | `robot_common_interface/msg/RecordMp4Status` | `/robot_camera` | `/robot_remote` |
| `/robot_camera/take_photo` | `robot_common_interface/msg/RecordPhotoRequest` | `/robot_remote` | `/robot_camera` |
| `/robot_camera/take_photo_ack` | `robot_common_interface/msg/RecordPhotoAck` | `/robot_camera` | `/robot_remote` |
| `/robot_hal/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/robot_hal` | — |
| `/robot_manager/robot_status` | `robot_common_interface/msg/RobotStatus` | `/robot_manager` | `/joint_shm_controller`、`/led_controller`、`/robot_remote` |
| `/robot_monitor/diagnostic_status` | `robot_common_interface/msg/RobotDiagnosticStatus` | `/robot_diagnostic_manager` | `/led_controller`、`/robot_manager`、`/robot_remote` |
| `/robot_monitor/power_on_status` | `robot_common_interface/msg/PowerOnStatus` | `/robot_monitor` | `/led_controller`、`/robot_remote` |
| `/robot_remote/knee_mode` | `std_msgs/msg/String` | `/robot_remote` | — |
| `/robot_remote/locked_state` | `std_msgs/msg/Bool` | `/robot_remote` | `/led_controller` |
| `/robot_remote/obstacle_avoidance` | `std_msgs/msg/Bool` | `/robot_remote` | — |
| `/robot_remote/reverse_head_tail` | `std_msgs/msg/String` | `/robot_remote` | `/led_controller` |
| `/robot_remote/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/robot_remote` | `/launch_ros_2132` |
| `/robot_roamerx/alg_dtc` | `robot_common_interface/msg/RoamerxAlgDtc` | `/robot_roamerx` | `/led_controller` |
| `/robot_roamerx/alg_status` | `robot_common_interface/msg/RoamerxAlgStatus` | `/robot_roamerx` | `/led_controller` |
| `/robot_roamerx/is_in_nav_control` | `std_msgs/msg/Bool` | `/robot_roamerx` | `/robot_manager` |
| `/robot_roamerx/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/robot_roamerx` | — |
| `/rosout` | `rcl_interfaces/msg/Log` | `/ComponentManager`、`/battery_controller`、`/bridge_image_topics_66`、`/controller_manager`、`/fill_light_controller`、`/imu_shm_publisher`、`/joint_shm_controller`、`/launch_ros_2132`、`/led_controller`、`/robot_camera`、`/robot_diagnostic_analyzer`、`/robot_diagnostic_manager`、`/robot_hal`、`/robot_manager`、`/robot_monitor`、`/robot_remote`、`/robot_roamerx`、`/robot_self_test_manager`、`/switch_controller`、`/zsi_actuator_driver`、`/zsi_imu_driver` | — |
| `/switch_controller/hw_estop/state` | `std_msgs/msg/Bool` | `/switch_controller` | `/robot_remote` |
| `/switch_controller/transition_event` | `lifecycle_msgs/msg/TransitionEvent` | `/switch_controller` | — |
| `/tf/tf_ack` | `robot_common_interface/msg/TFCardAck` | `/robot_camera` | `/robot_remote` |
| `/tf/tf_req` | `robot_common_interface/msg/TFCardRequest` | `/robot_remote` | `/robot_camera` |

### 每個節點的完整接線

**`/ComponentManager`**（其他）
- 發布：`/rosout`
- 訂閱：`/parameter_events`

**`/battery_controller`**（導航）
- 發布：`/battery_controller/battery1`、`/battery_controller/battery2`、`/battery_controller/battery_all`、`/battery_controller/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/bridge_image_topics_66`**（其他）
- 發布：`/rosout`
- 訂閱：`/front_camera/image_compressed`、`/parameter_events`、`/rear_camera/image_compressed`

**`/controller_manager`**（導航）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/fill_light_controller`**（導航）
- 發布：`/fill_light_controller/auto_work_state`、`/fill_light_controller/back_light/state`、`/fill_light_controller/front_light/state`、`/fill_light_controller/lux`、`/fill_light_controller/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/fill_light_controller/auto_work`、`/fill_light_controller/back_light/cmd`、`/fill_light_controller/front_light/cmd`、`/parameter_events`

**`/imu_shm_publisher`**（運控／HAL）
- 發布：`/imu_shm_publisher/imu_central`、`/imu_shm_publisher/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/joint_shm_controller`**（導航）
- 發布：`/joint_shm_controller/joint_cmd_echo`、`/joint_shm_controller/joint_sensor`、`/joint_shm_controller/joint_states`、`/joint_shm_controller/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`、`/robot_manager/robot_status`

**`/launch_ros_2132`**（其他）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/robot_remote/transition_event`

**`/led_controller`**（導航）
- 發布：`/led_controller/auto_work_state`、`/led_controller/transition_event`、`/parameter_events`、`/rosout`
- 訂閱：`/battery_controller/battery1`、`/battery_controller/battery2`、`/battery_controller/battery_all`、`/estop_controller/hw_estop/state`、`/estop_controller/sw_estop/state`、`/led_controller/auto_work`、`/led_controller/led_cmd`、`/parameter_events`、`/robot_manager/robot_status`、`/robot_monitor/diagnostic_status`、`/robot_monitor/power_on_status`、`/robot_remote/locked_state`、`/robot_remote/reverse_head_tail`、`/robot_roamerx/alg_dtc`、`/robot_roamerx/alg_status`

**`/robot_camera`**（感測驅動）
- 發布：`/front_camera/image_compressed`、`/parameter_events`、`/rear_camera/image_compressed`、`/robot_camera/record_mp4_ack`、`/robot_camera/record_mp4_status`、`/robot_camera/take_photo_ack`、`/rosout`、`/tf/tf_ack`
- 訂閱：`/parameter_events`、`/robot_camera/record_mp4`、`/robot_camera/take_photo`、`/tf/tf_req`

**`/robot_diagnostic_analyzer`**（其他）
- 發布：`/diagnostics_agg`、`/diagnostics_toplevel_state`、`/parameter_events`、`/rosout`
- 訂閱：`/diagnostics`、`/parameter_events`

**`/robot_diagnostic_manager`**（其他）
- 發布：`/parameter_events`、`/robot_monitor/diagnostic_status`、`/rosout`
- 訂閱：`/diagnostics_agg`、`/parameter_events`

**`/robot_hal`**（運控／HAL）
- 發布：`/diagnostics`、`/parameter_events`、`/robot_hal/transition_event`、`/rosout`
- 訂閱：`/parameter_events`

**`/robot_manager`**（運控／HAL）
- 發布：`/diagnostics`、`/parameter_events`、`/robot_manager/robot_status`、`/rosout`
- 訂閱：`/parameter_events`、`/robot_monitor/diagnostic_status`、`/robot_roamerx/is_in_nav_control`

**`/robot_monitor`**（運控／HAL）
- 發布：`/diagnostics`、`/parameter_events`、`/robot_monitor/power_on_status`、`/rosout`
- 訂閱：`/parameter_events`

**`/robot_remote`**（運控／HAL）
- 發布：`/diagnostics`、`/estop_controller/sw_estop/state`、`/fill_light_controller/auto_work`、`/fill_light_controller/back_light/cmd`、`/fill_light_controller/front_light/cmd`、`/parameter_events`、`/robot_camera/record_mp4`、`/robot_camera/take_photo`、`/robot_remote/knee_mode`、`/robot_remote/locked_state`、`/robot_remote/obstacle_avoidance`、`/robot_remote/reverse_head_tail`、`/robot_remote/transition_event`、`/rosout`、`/tf/tf_req`
- 訂閱：`/arc/charging_dock_info`、`/battery_controller/battery1`、`/battery_controller/battery2`、`/diagnostics_agg`、`/fill_light_controller/auto_work_state`、`/fill_light_controller/back_light/state`、`/fill_light_controller/front_light/state`、`/fill_light_controller/lux`、`/imu_shm_publisher/imu_central`、`/joint_shm_controller/joint_sensor`、`/parameter_events`、`/robot_camera/record_mp4_ack`、`/robot_camera/record_mp4_status`、`/robot_camera/take_photo_ack`、`/robot_manager/robot_status`、`/robot_monitor/diagnostic_status`、`/robot_monitor/power_on_status`、`/switch_controller/hw_estop/state`、`/tf/tf_ack`

**`/robot_roamerx`**（其他）
- 發布：`/arc/charging_dock_info`、`/diagnostics`、`/parameter_events`、`/robot_roamerx/alg_dtc`、`/robot_roamerx/alg_status`、`/robot_roamerx/is_in_nav_control`、`/robot_roamerx/transition_event`、`/rosout`
- 訂閱：`/battery_controller/battery1`、`/battery_controller/battery2`、`/cmd_vel`、`/control_right/test`、`/imu_shm_publisher/imu_central`、`/parameter_events`

**`/robot_self_test_manager`**（其他）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/switch_controller`**（導航）
- 發布：`/parameter_events`、`/rosout`、`/switch_controller/hw_estop/state`、`/switch_controller/transition_event`
- 訂閱：`/parameter_events`

**`/zsi_actuator_driver`**（其他）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

**`/zsi_imu_driver`**（感測驅動）
- 發布：`/parameter_events`、`/rosout`
- 訂閱：`/parameter_events`

