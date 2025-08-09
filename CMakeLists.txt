cmake_minimum_required(VERSION 3.10)
project(StereoVisualOdometry)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV 4.0 REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

add_executable(stereo_vo
    src/main.cpp
    src/feature_utils.cpp
)

target_link_libraries(stereo_vo ${OpenCV_LIBS})
