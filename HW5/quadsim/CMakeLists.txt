cmake_minimum_required(VERSION 3.0)
project(quadsim)

add_compile_options(-std=c++11)
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DCMAKE_PREFIX_PATH=/usr/local)

find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    rosflight_msgs
    std_msgs
    nav_msgs
    sensor_msgs
    geometry_msgs
    rosbag
    roslib
    cv_bridge
    message_generation
)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV Version: ${OpenCV_VERSION} (${OpenCV_DIR})")

add_message_files(
    FILES
    ImageData.msg
)

generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
)

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES quadsim
   CATKIN_DEPENDS rosflight_holodeck rosflight_msgs rosflight_firmware
#  DEPENDS system_lib
)

include_directories(
# include
# ${catkin_INCLUDE_DIRS}
)
