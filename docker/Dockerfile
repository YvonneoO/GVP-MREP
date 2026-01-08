# FROM gitlab.cerlab.lan.cmu.edu:5050/roboteam/isaac_sim:4.0.0-ros1
FROM qqyang/ros_noetic_planner:latest
# Install tools to add the ROS apt repository and its GPG key
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gnome-terminal \
      curl gnupg2 lsb-release && \
    # Import ROS GPG key & update
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    # Add ROS Noetic package repository
    echo "deb [arch=amd64] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" \
      > /etc/apt/sources.list.d/ros-latest.list && \
    # Update package index and install required packages
    apt-get update && \
    apt-get install -y --no-install-recommends \
      libtool \
      python3-rosdep \
      libgflags-dev \
      ros-noetic-mavros-msgs \
      ros-noetic-eigen-stl-containers \
      ros-noetic-rviz-visual-tools \
      ros-noetic-joy \
      ros-noetic-octomap-ros \
      python3-wstool \
      python3-catkin-tools \
      protobuf-compiler \
      libgoogle-glog-dev \
      ros-noetic-control-toolbox \
      ros-noetic-mavros \
      libzmqpp-dev \
      ros-noetic-catkin \
      git && \
    # Clean up apt caches to reduce image size
    rm -rf /var/lib/apt/lists/*

RUN sudo apt remove ros-noetic-mavlink -y

WORKDIR /root
RUN mkdir -p rotors-ws/src && cd rotors-ws && catkin init

RUN cd rotors-ws/src && \
    git clone https://github.com/NKU-UAVTeam/rotors-modified.git && \
    git clone https://github.com/ethz-asl/mav_comm.git && \
    git clone https://github.com/catkin/catkin_simple.git

RUN cd rotors-ws && \
    bash -c "source /opt/ros/noetic/setup.bash && catkin_make " && \
    echo "source ~/rotors-ws/devel/setup.bash --extend" >> ~/.bashrc

RUN apt-get update && \
    apt-get -y install ros-noetic-mavlink && \
    echo "source ~/rotors-ws/devel/setup.bash --extend" >> ~/.bashrc  
    

RUN mkdir -p catkin_ws/src && \ 
    cd catkin_ws/src && \
    git clone https://github.com/ethz-asl/gflags_catkin.git && \
    git clone https://github.com/ethz-asl/glog_catkin.git && \
    git clone https://github.com/catkin/catkin_simple.git && \
    # git clone https://github.com/NKU-MobFly-Robotics/GVP-MREP.git && \
    sed -i '1s/VERSION 2.8.3/VERSION 3.10/' glog_catkin/CMakeLists.txt && \
    sed -i 's/cmake_policy(SET CMP0053 OLD)/cmake_policy(SET CMP0053 NEW)/' glog_catkin/CMakeLists.txt && \
    sed -i '1s/VERSION 2.8.3/VERSION 3.10/' gflags_catkin/CMakeLists.txt && \
    sed -i '22a -DCMAKE_POLICY_VERSION_MINIMUM=3.5' gflags_catkin/CMakeLists.txt

        
# Copy and make the entrypoint script executable
COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["entrypoint.sh"]
CMD ["/bin/bash"]

