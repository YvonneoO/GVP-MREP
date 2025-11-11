cmake_minimum_required(VERSION 3.0.2)
project(traj_exc)

add_definitions(-std=c++14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cs_add_executable(traj_exc_node src/traj_exc_node.cpp)

target_link_libraries(traj_exc_node
    ${catkin_LIBRARIES})

cs_add_executable(ugv_traj_exec src/ugv_traj_exec.cpp)

target_link_libraries(ugv_traj_exec
    ${catkin_LIBRARIES})
##########
# EXPORT #
##########
cs_install()
cs_export()
