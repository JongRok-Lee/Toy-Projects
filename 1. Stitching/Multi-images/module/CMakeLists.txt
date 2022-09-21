cmake_minimum_required(VERSION 3.10)
project(module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MODULE_SOURCE_FILES src/extractor.cpp)

# find_package(OpenCV 4.5 REQUIRED HINTS ~/workspace/stitching/opencv-4.5.5/install/lib/cmake/opencv4)
find_package(OpenCV 3.2 REQUIRED)

add_library(module ${MODULE_SOURCE_FILES})

target_include_directories(module PUBLIC
                            include
                            ${OpenCV_INCLUDE_DIRS}
                            )

target_link_libraries(module PUBLIC
                        ${OpenCV_LIBS})