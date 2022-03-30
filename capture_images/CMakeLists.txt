cmake_minimum_required(VERSION 3.21)
project(capture_images)

find_package(Pylon QUIET)
if (NOT ${Pylon_FOUND})
    include(/opt/pylon/share/cmake/FindPylon.cmake)
endif()
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} ${Pylon_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 20)

add_executable(capture_images main.cpp)

target_link_libraries(capture_images ${OpenCV_LIBS} ${Pylon_LIBRARIES})
