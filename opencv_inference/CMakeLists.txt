cmake_minimum_required(VERSION 3.21)
project(opencv_inference)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencv_inference main.cpp)

target_link_libraries(opencv_inference ${OpenCV_LIBS})
