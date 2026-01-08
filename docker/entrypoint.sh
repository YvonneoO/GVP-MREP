#!/bin/bash
set -e

echo "Starting entrypoint..."

# Source ROS setup
source /opt/ros/noetic/setup.bash
source /root/rotors-ws/devel/setup.bash


echo "Copying required packages..."
cp -r /root/catkin_ws/src/gflags_catkin /root/GVP-MREP/src && echo "✓ gflags_catkin copied"
cp -r /root/catkin_ws/src/glog_catkin /root/GVP-MREP/src && echo "✓ glog_catkin copied"
cp -r /root/catkin_ws/src/catkin_simple /root/GVP-MREP/src && echo "✓ catkin_simple copied"
# Check if mounted workspace exists
if [ -d /root/GVP-MREP/src ]; then
  echo "Mounted workspace found at /root/GVP-MREP"
  cd /root/GVP-MREP
  catkin_make && source devel/setup.bash
else
  echo "/root/GVP-MREP/src does not exist. Please mount your workspace correctly."
fi

echo "Entrypoint complete. Starting bash shell..."
exec /bin/bash