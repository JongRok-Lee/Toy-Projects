cmake_minimum_required(VERSION 3.10)
project(module LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(MODULE_SOURCE_FILES src/module.cpp)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
set(Eigen3_LIBS Eigen3::Eigen)

add_library(module ${MODULE_SOURCE_FILES})

target_include_directories(module PUBLIC
                            include
                            ${Eigen3_INCLUDE_DIRS}
                            ${OpenCV_INCLUDE_DIRS}
                            )

target_link_libraries(module PUBLIC
                        ${Eigen3_LIBS}
                        ${OpenCV_LIBS}
                        )