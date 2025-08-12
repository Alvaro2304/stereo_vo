cmake_minimum_required(VERSION 3.10)
project(StereoVisualOdometry)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV 4.0 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(src)

add_executable(stereo_vo
    src/main.cpp
    src/calibration.cpp
    src/feature_bucket.cpp
)

target_link_libraries(stereo_vo ${OpenCV_LIBS})
