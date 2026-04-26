
ur_robot_driver 启动真实 UR10e，推荐这样开。


conda deactivate
source /opt/ros/jazzy/setup.bash
source ~/ur_10e/install/setup.bash
然后启动 UR 驱动，把 192.168.x.x 换成你的机器人 IP：

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:=192.168.x.x \
  launch_rviz:=false


ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur10e \
  robot_ip:=192.168.x.x \
  reverse_ip:=192.168.x.y \
  launch_rviz:=false
机器人示教器上也要做一件事：安装并运行 External Control 程序。URCap 文件在这里：

/opt/ros/jazzy/share/ur_robot_driver/resources/externalcontrol-1.0.5.urcap
启动顺序建议：

电脑能 ping 通机器人：
ping 192.168.x.x
启动 driver：
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur10e robot_ip:=机器人IP launch_rviz:=false
在示教器上运行 External Control 程序。

另开终端启动你的 MoveIt：

conda deactivate
source ~/ur_10e/install/setup.bash
ros2 launch ur10e_moveit ur_moveit.launch.py
检查控制器：
ros2 control list_controllers
你需要看到：

scaled_joint_trajectory_controller active
joint_state_broadcaster active


