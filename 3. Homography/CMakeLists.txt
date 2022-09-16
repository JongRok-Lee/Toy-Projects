cmake_minimum_required(VERSION 3.10)
Project(ransac LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(module)

add_executable(ransac example/main.cpp)
target_link_libraries(ransac PUBLIC
                        module)