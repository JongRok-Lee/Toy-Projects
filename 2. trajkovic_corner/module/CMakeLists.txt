cmake_minimum_required(VERSION 3.10)
project(module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MODULE_SOURCE_FILES src/detection.cpp)

find_package(OpenCV REQUIRED)

add_library(module ${MODULE_SOURCE_FILES})

target_include_directories(module PUBLIC
                            include
                            ${OpenCV_INCLUDE_DIRS}
                            )

target_link_libraries(module PUBLIC
                        ${OpenCV_LIBS})