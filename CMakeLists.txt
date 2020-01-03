cmake_minimum_required(VERSION 3.10.2)
project(pobr)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR /home/mateusz/lib/opencv/installation/OpenCV-3.4.4/share/OpenCV/)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(pobr main.cpp)

target_link_libraries(pobr ${OpenCV_LIBS})