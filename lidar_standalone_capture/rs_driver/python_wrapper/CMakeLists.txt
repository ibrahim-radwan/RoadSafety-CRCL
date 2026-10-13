cmake_minimum_required(VERSION 3.5)

project(rs_driver_python_wrapper)

message(=============================================================)
message("-- Ready to compile Python wrapper bridge")
message(=============================================================)

if (${ENABLE_PCL_POINTCLOUD})
  find_package(PCL REQUIRED)
  find_package(yaml-cpp REQUIRED)
  include_directories(${PCL_INCLUDE_DIRS})
  link_directories(${PCL_LIBRARY_DIRS})
  add_definitions(${PCL_DEFINITIONS})
endif (${ENABLE_PCL_POINTCLOUD})

include_directories(${DRIVER_INCLUDE_DIRS})

add_executable(demo_online_bridge
              demo_online_bridge.cpp)

target_link_libraries(demo_online_bridge
                    ${EXTERNAL_LIBS}
                    yaml-cpp)
