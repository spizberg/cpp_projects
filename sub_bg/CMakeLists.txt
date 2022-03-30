cmake_minimum_required(VERSION 3.21)
project(sub_bg)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(sub_bg main.cpp)

target_link_libraries(sub_bg ${OpenCV_LIBS})
