cmake_minimum_required(VERSION 3.12)
project(robosense_lidar_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(
  COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_RESULT
)

if(NOT PYBIND11_RESULT EQUAL 0)
  message(FATAL_ERROR
    "pybind11 is not installed for ${Python3_EXECUTABLE}. "
    "Install it with: ${Python3_EXECUTABLE} -m pip install pybind11")
endif()

find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(robosense_lidar robosense_lidar.cpp)

find_package(yaml-cpp REQUIRED)

target_include_directories(robosense_lidar PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
)

# rs_driver is header-only/template-based in this release. Online UDP mode
# needs pthread and does not require libpcap.
target_link_libraries(
    robosense_lidar
    PRIVATE
    yaml-cpp
    pcap
    pthread)