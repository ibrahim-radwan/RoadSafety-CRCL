/*********************************************************************************************************************
Copyright (c) 2020 RoboSense
All rights reserved

By downloading, copying, installing or using the software you agree to this license. If you do not agree to this
license, do not download, install, copy or use the software.

License Agreement
For RoboSense LiDAR SDK Library
(3-clause BSD License)

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following
disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following
disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the names of the RoboSense, nor Suteng Innovation Technology, nor the names of other contributors may be used
to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*********************************************************************************************************************/

#include <rs_driver/api/lidar_driver.hpp>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#ifdef ENABLE_PCL_POINTCLOUD
#include <rs_driver/msg/pcl_point_cloud_msg.hpp>
#else
#include <rs_driver/msg/point_cloud_msg.hpp>
#endif

using namespace robosense::lidar;

SyncQueue<std::shared_ptr<PointCloudT<PointXYZI>>> free_cloud_queue;
SyncQueue<std::shared_ptr<PointCloudT<PointXYZI>>> stuffed_cloud_queue;
SyncQueue<std::shared_ptr<ImuData>> free_imu_data_queue;
SyncQueue<std::shared_ptr<ImuData>> stuffed_imu_data_queue;

bool to_exit_process = false;

struct Config {
  std::string lidar_type;
  int msop_port;
  int difop_port;
  int imu_port;
  int user_layer_bytes;
  int tail_layer_bytes;
  float min_distance;
  float max_distance;
  bool use_lidar_clock;
  bool dense_points;
  bool ts_first_point;
  float start_angle;
  float end_angle;
  std::string lidar_ip;
  std::string host_ip;
};

Config loadConfig(const std::string& filename) {
  YAML::Node node = YAML::LoadFile(filename);
  Config config;

  config.lidar_type = node["lidar"]["lidar_type"].as<std::string>();
  config.msop_port = node["lidar"]["msop_port"].as<int>();
  config.difop_port = node["lidar"]["difop_port"].as<int>();
  config.imu_port = node["lidar"]["imu_port"].as<int>();
  config.user_layer_bytes = node["lidar"]["user_layer_bytes"].as<int>();
  config.tail_layer_bytes = node["lidar"]["tail_layer_bytes"].as<int>();
  config.min_distance = node["lidar"]["min_distance"].as<float>();
  config.max_distance = node["lidar"]["max_distance"].as<float>();
  config.use_lidar_clock = node["lidar"]["use_lidar_clock"].as<bool>();
  config.dense_points = node["lidar"]["dense_points"].as<bool>();
  config.ts_first_point = node["lidar"]["ts_first_point"].as<bool>();
  config.start_angle = node["lidar"]["start_angle"].as<float>();
  config.end_angle = node["lidar"]["end_angle"].as<float>();
  config.lidar_ip = node["network"]["lidar_ip"].as<std::string>();
  config.host_ip = node["network"]["host_ip"].as<std::string>();
  return config;
}

RSDriverParam loadParams(const Config& config) {
  RSDriverParam param;

  std::string lidar_type = config.lidar_type;
  if (lidar_type == "RSAIRY") {
    param.lidar_type = LidarType::RSAIRY;
  } else if (lidar_type == "RS16") {
    param.lidar_type = LidarType::RS16;
  } else if (lidar_type == "RS32") {
    param.lidar_type = LidarType::RS32;
  } else if (lidar_type == "RSBP") {
    param.lidar_type = LidarType::RSBP;
  } else if (lidar_type == "RSAIRYLITE_ETH") {
    param.lidar_type = LidarType::RSAIRYLITE_ETH;
  } else if (lidar_type == "RSFAIRY") {
    param.lidar_type = LidarType::RSFAIRY;
  } else if (lidar_type == "RSHELIOS") {
    param.lidar_type = LidarType::RSHELIOS;
  } else if (lidar_type == "RSHELIOS_16P") {
    param.lidar_type = LidarType::RSHELIOS_16P;
  } else if (lidar_type == "RS128") {
    param.lidar_type = LidarType::RS128;
  } else if (lidar_type == "RSP80") {
    param.lidar_type = LidarType::RSP80;
  } else {
    throw std::runtime_error("Unsupported LiDAR type: " + lidar_type);
  }

  param.input_type = InputType::ONLINE_LIDAR;
  param.input_param.host_address = config.host_ip;
  param.input_param.msop_port = config.msop_port;
  param.input_param.difop_port = config.difop_port;
  param.input_param.imu_port = config.imu_port;
  param.input_param.user_layer_bytes = config.user_layer_bytes;
  param.input_param.tail_layer_bytes = config.tail_layer_bytes;
  param.decoder_param.min_distance = config.min_distance;
  param.decoder_param.max_distance = config.max_distance;
  param.decoder_param.use_lidar_clock = config.use_lidar_clock;
  param.decoder_param.dense_points = config.dense_points;
  param.decoder_param.ts_first_point = config.ts_first_point;
  param.decoder_param.start_angle = config.start_angle;
  param.decoder_param.end_angle = config.end_angle;
  return param;
}

std::shared_ptr<PointCloudT<PointXYZI>> driverGetPointCloudFromCallerCallback(void) {
  std::shared_ptr<PointCloudT<PointXYZI>> msg = free_cloud_queue.pop();
  if (msg.get() != NULL) {
    return msg;
  }
  return std::make_shared<PointCloudT<PointXYZI>>();
}

void driverReturnPointCloudToCallerCallback(std::shared_ptr<PointCloudT<PointXYZI>> msg) {
  stuffed_cloud_queue.push(msg);
}

std::shared_ptr<ImuData> driverGetIMUDataFromCallerCallback(void) {
  std::shared_ptr<ImuData> msg = free_imu_data_queue.pop();
  if (msg.get() != NULL) {
    return msg;
  }
  return std::make_shared<ImuData>();
}

void driverReturnImuDataToCallerCallback(const std::shared_ptr<ImuData>& msg) {
  stuffed_imu_data_queue.push(msg);
}

void exceptionCallback(const Error& code) {
  RS_WARNING << code.toString() << RS_REND;
}

void emitPointCloudJSON(const std::shared_ptr<PointCloudT<PointXYZI>>& msg) {
  std::ostringstream oss;
  oss << "{\"type\":\"pointcloud\",\"seq\":" << msg->seq << ",\"timestamp\":" << msg->timestamp
      << ",\"points\":[";

  for (size_t i = 0; i < msg->points.size(); ++i) {
    const auto& p = msg->points[i];
    if (i > 0) {
      oss << ",";
    }
    oss << "[" << p.x << "," << p.y << "," << p.z << "," << static_cast<int>(p.intensity) << "]";
  }
  oss << "]}";

  std::cout << oss.str() << std::endl;
}

void emitImuJSON(const std::shared_ptr<ImuData>& msg) {
  std::ostringstream oss;
  oss << "{\"type\":\"imu\",\"timestamp\":" << msg->timestamp
      << ",\"angular_velocity\":[" << msg->angular_velocity_x << "," << msg->angular_velocity_y << "," << msg->angular_velocity_z
      << "],\"linear_acceleration\":[" << msg->linear_acceleration_x << "," << msg->linear_acceleration_y << "," << msg->linear_acceleration_z
      << "],\"state\":" << (msg->state ? 1 : 0) << "}";
  std::cout << oss.str() << std::endl;
}

void processCloud(void) {
  while (!to_exit_process) {
    std::shared_ptr<PointCloudT<PointXYZI>> msg = stuffed_cloud_queue.popWait();
    if (msg.get() == NULL) {
      continue;
    }
    emitPointCloudJSON(msg);
    free_cloud_queue.push(msg);
  }
}

void processImuData(void) {
  while (!to_exit_process) {
    std::shared_ptr<ImuData> msg = stuffed_imu_data_queue.popWait();
    if (msg.get() == NULL) {
      continue;
    }
    emitImuJSON(msg);
    free_imu_data_queue.push(msg);
  }
}

int main(int argc, char* argv[]) {
  try {
    Config config = loadConfig("../config/config.yaml");
    RSDriverParam param = loadParams(config);

    LidarDriver<PointCloudT<PointXYZI>> driver;
    driver.regPointCloudCallback(driverGetPointCloudFromCallerCallback, driverReturnPointCloudToCallerCallback);
    driver.regExceptionCallback(exceptionCallback);
    driver.regImuDataCallback(driverGetIMUDataFromCallerCallback, driverReturnImuDataToCallerCallback);

    if (!driver.init(param)) {
      std::cerr << "Driver Initialize Error..." << std::endl;
      return -1;
    }

    std::thread cloud_handle_thread = std::thread(processCloud);
    std::thread imuData_handle_thread = std::thread(processImuData);
    driver.start();

    while (true) {
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    to_exit_process = true;
    cloud_handle_thread.join();
    imuData_handle_thread.join();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Bridge error: " << e.what() << std::endl;
    return 1;
  }
}
