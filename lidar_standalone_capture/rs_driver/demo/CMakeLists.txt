cmake_minimum_required(VERSION 3.16)

project(robosense_lidar_python)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Explicitly use Python 3.8
set(Python3_EXECUTABLE /usr/bin/python)

find_package(Python3 3.8 EXACT REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 REQUIRED)
find_package(yaml-cpp REQUIRED)

set(RS_DRIVER_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../rs_driver)

pybind11_add_module(robosense_lidar robosense_lidar.cpp)

target_include_directories(
    robosense_lidar
    PRIVATE
    ${RS_DRIVER_DIR}/src
)

target_link_libraries(
    robosense_lidar
    PRIVATE
    yaml-cpp
    pthread
)