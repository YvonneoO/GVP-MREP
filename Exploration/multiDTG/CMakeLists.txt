cmake_minimum_required(VERSION 3.0.2)
project(multiDTG)

add_definitions(-std=c++14)
set(CMAKE_BUILD_TYPE Release)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

cs_add_library(${PROJECT_NAME} src/multiDTG.cpp src/multiDTG_path_search.cpp)

cs_add_executable(dtg_recorder_node src/dtg_recorder_node.cpp)
target_link_libraries(dtg_recorder_node ${catkin_LIBRARIES})

cs_add_executable(dtg_reader_node src/dtg_reader_node.cpp)
target_link_libraries(dtg_reader_node ${catkin_LIBRARIES})
##########
# EXPORT #
##########
cs_install()
cs_export()