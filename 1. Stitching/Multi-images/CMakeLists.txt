cmake_minimum_required(VERSION 3.10)
Project(stitching LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(module)

add_executable(run_stitching example/main.cpp)
add_executable(opencv_api example/api.cpp)
target_link_libraries(run_stitching PUBLIC
                        module
                        stdc++fs)
target_link_libraries(opencv_api PUBLIC
                        module
                        stdc++fs)