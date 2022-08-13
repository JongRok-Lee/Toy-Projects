cmake_minimum_required(VERSION 3.10)
Project(corner_detection LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(module)

add_executable(corner_detection example/main.cpp)
target_link_libraries(corner_detection PUBLIC
                        module)