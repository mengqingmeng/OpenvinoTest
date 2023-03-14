cmake_minimum_required(VERSION 3.24)
set(CMAKE_CXX_STANDARD  17)
project(OpenvinoTest)

#OPENVINO
set(CMAKE_PREFIX_PATH "/Users/mqm/WORKSPACE/OPENVINO/openvino/build")
find_package(OpenVINO REQUIRED)

# OPENCV
set(CMAKE_PREFIX_PATH "/Users/mqm/WORKSPACE/OPENCV/opencv-4.7.0/build")
find_package(OpenCV REQUIRED)

add_executable(OpenvinoTest main.cpp)

target_link_libraries(OpenvinoTest PRIVATE openvino::runtime ${OpenCV_LIBS})
