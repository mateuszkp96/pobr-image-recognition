cmake_minimum_required(VERSION 3.10.2)
project(pobr)
set(CMAKE_CXX_STANDARD 14)

set(OpenCV_DIR /home/mateusz/lib/opencv/installation/OpenCV-3.4.4/share/OpenCV/)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(HEADER_FILES
        Constants.h
        ImageUtils.h
        ObjectFeatures.h
        Processor.h
        Utils.h
        )

set(SOURCE_FILES
        ImageUtils.cpp
        Utils.cpp
        main.cpp
        ObjectFeatures.cpp
        Processor.cpp
        )

add_executable(pobr ${SOURCE_FILES} ${HEADER_FILES})

target_link_libraries(pobr ${OpenCV_LIBS})