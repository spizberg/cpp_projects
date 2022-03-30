cmake_minimum_required(VERSION 3.21)
project(torch_test)

set(Torch_DIR /usr/local/include/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 20)

add_executable(torch_test main.cpp)

target_link_libraries(torch_test ${OpenCV_LIBS})
target_link_libraries(torch_test ${TORCH_LIBRARIES})
