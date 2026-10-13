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
#include <pcl/point_types.h>

#include <iostream> 
#include <string>
#include <yaml-cpp/yaml.h>

#ifdef ENABLE_PCL_POINTCLOUD
#include <rs_driver/msg/pcl_point_cloud_msg.hpp>
#else
#include <rs_driver/msg/point_cloud_msg.hpp>
#endif

//#define ORDERLY_EXIT

// Define the macro: 1 to enable IMU parsing, 0 to disable IMU parsing
#define ENABLE_IMU_PARSE 1 

typedef PointXYZI PointT;
typedef PointCloudT<PointT> PointCloudMsg;

using namespace robosense::lidar;

//All the queues
SyncQueue<std::shared_ptr<PointCloudMsg>> free_cloud_queue;
SyncQueue<std::shared_ptr<PointCloudMsg>> stuffed_cloud_queue;

SyncQueue<std::shared_ptr<ImuData>> free_imu_data_queue;
SyncQueue<std::shared_ptr<ImuData>> stuffed_imu_data_queue;

struct Config 
{ 
  // LiDAR 
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
  

  // Network 
  std::string lidar_ip; 
  std::string host_ip; 

  //Calibration
  double calibration_x;
  double calibration_y;
  double calibration_z;
};

//Adding Gyro Bias
struct GyroBias
{
    double x=0.0147152;
    double y=-0.0116165;
    double z=-0.00163704;
};
GyroBias gyro_bias;
//#define ENABLE_CALIBRATION 
#ifdef ENABLE_CALIBRATION
    bool calibrating_gyro = true;
    uint32_t calibration_count = 0;
    const uint32_t CALIBRATION_SAMPLES = 5000;
    double gyro_sum_x = 0.0;
    double gyro_sum_y = 0.0;
    double gyro_sum_z = 0.0;
#endif

//Adding IMU state
struct IMUState
{
    //Raw values
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double gx = 0.0;
    double gy = 0.0;
    double gz = 0.0;
    
    // Orientation
    double roll  = 0.0;
    double pitch = 0.0;
    double yaw   = 0.0;

    // Velocity
    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double velocity_z = 0.0;

    // Previous timestamp
    double last_timestamp = 0.0;

    bool initialized = false;
    double previous_timestamp = 0.0;
};
IMUState imu_state;

const double GRAVITY = 9.80665;
const double ACCEL_FILTER_ALPHA = 0.98;

void updateImuState(
    double timestamp,
    double ax, double ay, double az,
    double gx, double gy, double gz)
{
  // First IMU sample
  if (!imu_state.initialized)
  {
    imu_state.previous_timestamp = timestamp;

    imu_state.roll =
        std::atan2(ay, az);

    imu_state.pitch =
        std::atan2(
            -ax,
            std::sqrt(ay * ay + az * az));

    imu_state.yaw = 0.0;

    imu_state.velocity_x = 0.0;
    imu_state.velocity_y = 0.0;
    imu_state.velocity_z = 0.0;

    imu_state.initialized = true;

    return;
  }

  double dt = timestamp - imu_state.previous_timestamp;

  imu_state.previous_timestamp = timestamp;

  // Protect against invalid timestamps
  if (dt <= 0.0 || dt > 1.0)
  {
    return;
  }

  double roll_gyro =
      imu_state.roll + gx * dt;
  double pitch_gyro =
      imu_state.pitch + gy * dt;
  double yaw_gyro =
      imu_state.yaw + gz * dt;


  // Calculate roll/pitch from accelerometer
  double roll_acc =
      std::atan2(ay, az);
  double pitch_acc =
      std::atan2(
          -ax,
          std::sqrt(ay * ay + az * az));

  imu_state.ax = ax;
  imu_state.ay = ay;
  imu_state.az = az;
  imu_state.gx = gx;
  imu_state.gy = gy;
  imu_state.gz = gz;
  
  imu_state.roll =
      ACCEL_FILTER_ALPHA * roll_gyro +
      (1.0 - ACCEL_FILTER_ALPHA) * roll_acc;

  imu_state.pitch =
      ACCEL_FILTER_ALPHA * pitch_gyro +
      (1.0 - ACCEL_FILTER_ALPHA) * pitch_acc;

  imu_state.yaw = yaw_gyro;


  // Convert body acceleration to world frame
  double cr = std::cos(imu_state.roll);
  double sr = std::sin(imu_state.roll);
  double cp = std::cos(imu_state.pitch);
  double sp = std::sin(imu_state.pitch);
  double cy = std::cos(imu_state.yaw);
  double sy = std::sin(imu_state.yaw);


  // Rotation matrix:
  //
  // R = Rz(yaw) * Ry(pitch) * Rx(roll)
  //

  double r11 = cy * cp;
  double r12 = cy * sp * sr - sy * cr;
  double r13 = cy * sp * cr + sy * sr;

  double r21 = sy * cp;
  double r22 = sy * sp * sr + cy * cr;
  double r23 = sy * sp * cr - cy * sr;

  double r31 = -sp;
  double r32 = cp * sr;
  double r33 = cp * cr;


  // Body -> world
  double world_ax =
      r11 * ax +
      r12 * ay +
      r13 * az;

  double world_ay =
      r21 * ax +
      r22 * ay +
      r23 * az;

  double world_az =
      r31 * ax +
      r32 * ay +
      r33 * az;

  // Remove gravity
  world_az -= GRAVITY;

  // Integrate acceleration -> velocity
  imu_state.velocity_x += world_ax * dt;
  imu_state.velocity_y += world_ay * dt;
  imu_state.velocity_z += world_az * dt;
}

//
// @brief point cloud callback function. The caller should register it to the lidar driver.
//        Via this fucntion, the driver gets an free/unused point cloud message from the caller.
// @param msg  The free/unused point cloud message.
//
std::shared_ptr<PointCloudMsg> driverGetPointCloudFromCallerCallback(void)
{
  // Note: This callback function runs in the packet-parsing/point-cloud-constructing thread of the driver,
  //       so please DO NOT do time-consuming task here.
  std::shared_ptr<PointCloudMsg> msg = free_cloud_queue.pop();
  if (msg.get() != NULL)
  {
    return msg;
  }

  return std::make_shared<PointCloudMsg>();
}

//
// @brief point cloud callback function. The caller should register it to the lidar driver.
//        Via this function, the driver gets/returns a stuffed point cloud message to the caller.
// @param msg  The stuffed point cloud message.
//
void driverReturnPointCloudToCallerCallback(std::shared_ptr<PointCloudMsg> msg)
{
  // Note: This callback function runs in the packet-parsing/point-cloud-constructing thread of the driver,
  //       so please DO NOT do time-consuming task here. Instead, process it in caller's own thread. (see processCloud()
  //       below)
  stuffed_cloud_queue.push(msg);
}

//
// @brief IMU data callback function. The caller should register it to the lidar driver.
//        Via this function, the driver gets/returns a stuffed IMU data message to the caller. 
// @param msg  The stuffed IMU data message.
//
std::shared_ptr<ImuData> driverGetIMUDataFromCallerCallback(void)
{
   // Note: This callback function runs in the packet-parsing/imu-data-constructing thread of the driver, 
  //       so please DO NOT do time-consuming task here.
  std::shared_ptr<ImuData> msg = free_imu_data_queue.pop();
  if (msg.get() != NULL)
  {
    return msg;
  }

  return std::make_shared<ImuData>();
}

//
// @brief IMU data callback function. The caller should register it to the lidar driver.
//        Via this function, the driver gets/returns a stuffed IMU data message to the caller. 
// @param msg  The stuffed IMU data message.
//
void driverReturnImuDataToCallerCallback(const std::shared_ptr<ImuData>& msg)
{
  // Note: This callback function runs in the packet-parsing/imu-data-constructing thread of the driver, 
  //       so please DO NOT do time-consuming task here. Instead, process it in caller's own thread. (see processImuData() below)
  stuffed_imu_data_queue.push(msg);
}

/******************************************************************************************************/
bool to_exit_process = false;
void processImuData(void)
{
  uint32_t imu_cnt = 0;
  while (!to_exit_process)
  {
    std::shared_ptr<ImuData> msg = stuffed_imu_data_queue.popWait();
    if (msg.get() == NULL)
    {
      continue;
    }
    double ax = msg->linear_acceleration_x;
    double ay = msg->linear_acceleration_y;
    double az = msg->linear_acceleration_z; 
    double gx = msg->angular_velocity_x;
    double gy = msg->angular_velocity_y;
    double gz = msg->angular_velocity_z;
    
#ifdef ENABLE_CALIBRATION
    if (calibrating_gyro){
        gyro_sum_x += gx;
        gyro_sum_y += gy;
        gyro_sum_z += gz;
        RS_MSG << "Gyro Calibration msg: sum_gx" << gyro_sum_x << " sum_gy: " << gyro_sum_y << " sum_gz: " << gyro_sum_z << RS_REND;
        
        calibration_count++;
        if(calibration_count>=CALIBRATION_SAMPLES)
        {
            gyro_bias.x = gyro_sum_x / CALIBRATION_SAMPLES;
            gyro_bias.y = gyro_sum_y / CALIBRATION_SAMPLES;
            gyro_bias.z = gyro_sum_z / CALIBRATION_SAMPLES;
            calibrating_gyro = false;
            RS_MSG << "Gyro Calibration msg: "
            << "\n\n======================================\n"
            << "Gyro Calibration completed\n"
            << "Bias X: " << gyro_bias.x
            << "Bias Y: " << gyro_bias.y
            << "Bias Z: " << gyro_bias.z
            << "\n\n======================================\n"
            << RS_REND;
        }
    }
#endif

    // Well, it is time to process the IMU data msg, even it is time-consuming.

    imu_cnt++;
#ifndef ENABLE_CALIBRATION
    gx -= gyro_bias.x;
    gy -= gyro_bias.y;
    gz -= gyro_bias.z;
#endif
    
    
    updateImuState(
        static_cast<double>(msg->timestamp),
        ax,
        ay,
        az,
        gx,
        gy,
        gz);
        
    double roll_deg =
        imu_state.roll * 180.0 / M_PI;

    double pitch_deg =
        imu_state.pitch * 180.0 / M_PI;

    double yaw_deg =
        imu_state.yaw * 180.0 / M_PI;


    // Keep yaw between -180 and +180 degrees
    while (yaw_deg > 180.0)
        yaw_deg -= 360.0;

    while (yaw_deg < -180.0)
        yaw_deg += 360.0;
        
    double vx = imu_state.velocity_x;
    double vy = imu_state.velocity_y;
    double vz = imu_state.velocity_z;
    double speed = std::sqrt(
        vx * vx +
        vy * vy +
        vz * vz);
    
#if 1
    if(imu_cnt % 200 == 0){
    RS_MSG << "msg: " << imu_cnt << " imu data ts: " <<std::dec<<std::to_string(msg->timestamp) << RS_REND;
    RS_DEBUG  <<"imu data: " << " , linear_a_x" << msg->linear_acceleration_x 
      << " , linear_a_y " << msg->linear_acceleration_y << "  , linear_a_z" << msg->linear_acceleration_z   
      << " , angular_v_x " << msg->angular_velocity_x << " , angular_v_y" << msg->angular_velocity_y 
      << " , angular_v_z" <<msg->angular_velocity_z << RS_REND;
    RS_DEBUG  <<"imu data: " << " , roll" << roll_deg << " , pitch " << pitch_deg << "  , yaw" << yaw_deg   
      << " , vx " << vx << " , vy" << vy << " , vz" << vz << " , \nspeed" << speed << RS_REND;
      }
#endif

    free_imu_data_queue.push(msg);
  }

}
//
// @brief exception callback function. The caller should register it to the lidar driver.
//        Via this function, the driver inform the caller that something happens.
// @param code The error code to represent the error/warning/information
//
void exceptionCallback(const Error& code)
{
  // Note: This callback function runs in the packet-receving and packet-parsing/point-cloud_constructing thread of the
  // driver,
  //       so please DO NOT do time-consuming task here.
  RS_WARNING << code.toString() << RS_REND;
}


void processCloud(void)
{
  while (!to_exit_process)
  {
    std::shared_ptr<PointCloudMsg> msg = stuffed_cloud_queue.popWait();
    if (msg.get() == NULL)
    {
      continue;
    }
#if 0
    // Well, it is time to process the point cloud msg, even it is time-consuming.
    RS_MSG << "msg: " << msg->seq << " point cloud size: " << msg->points.size() << RS_REND;


    for (auto it = msg->points.begin(); it != msg->points.end(); it++)
    {
      std::cout << std::fixed << std::setprecision(8) << "(" << it->x << ", " << it->y << ", " << it->z << ", "
                << (int)it->intensity << ")" << std::endl;
    }
#endif

    free_cloud_queue.push(msg);


  }
}

void processCloud_range_reflective(void)
{
  while (!to_exit_process)
  {
    std::shared_ptr<PointCloudMsg> msg = stuffed_cloud_queue.popWait();

    if (msg.get() == NULL)
    {
      continue;
    }
#if 0    
    RS_MSG << "msg: " << msg->seq << " pointcloud size: " << msg->points.size() << RS_REND;

    // Process each point in the point cloud
    for (const auto& point : msg->points)
    {
      float x = point.x;
      float y = point.y;
      float z = point.z;

      // Calculate distance from LiDAR to point
      float range = std::sqrt(x * x + y * y + z * z);

      // Reflectivity / intensity
      float reflectivity = point.intensity;

      std::cout << std::fixed << std::setprecision(3)
                << x << ", "
                << y << ", "
                << z << ", "
                << range << ", "
                << reflectivity
                << std::endl;
    }
#endif
    // Return point cloud object to free queue
    free_cloud_queue.push(msg);
  }
}

Config loadConfig(const std::string& filename)
{
  try{
    Config config;
    YAML::Node configNode = YAML::LoadFile(filename);

    // LiDAR
    config.lidar_type = configNode["lidar"]["lidar_type"].as<std::string>();
    config.msop_port = configNode["lidar"]["msop_port"].as<int>();
    config.difop_port = configNode["lidar"]["difop_port"].as<int>();
    config.imu_port = configNode["lidar"]["imu_port"].as<int>();
    config.user_layer_bytes = configNode["lidar"]["user_layer_bytes"].as<int>();
    config.tail_layer_bytes = configNode["lidar"]["tail_layer_bytes"].as<int>();

    config.min_distance = configNode["lidar"]["min_distance"].as<float>();
    config.max_distance = configNode["lidar"]["max_distance"].as<float>();

    config.use_lidar_clock = configNode["lidar"]["use_lidar_clock"].as<bool>();
    config.dense_points = configNode["lidar"]["dense_points"].as<bool>();
    config.ts_first_point = configNode["lidar"]["ts_first_point"].as<bool>();

    config.start_angle = configNode["lidar"]["start_angle"].as<float>();
    config.end_angle = configNode["lidar"]["end_angle"].as<float>();

    // Network
    config.lidar_ip = configNode["network"]["lidar_ip"].as<std::string>();
    config.host_ip = configNode["network"]["host_ip"].as<std::string>();

    return config;
  } catch (const std::exception& e) {
    std::cerr << "Error loading config file: " << e.what() << std::endl;
    throw; // Rethrow the exception to indicate failure 
    return Config(); // Return an empty config object in case of error
  }
}

RSDriverParam loadParams(Config config)
{
  RSDriverParam param;  ///< Create a parameter object

  std::string lidar_type = config.lidar_type;
  if (lidar_type == "RSAIRY")
  {
      param.lidar_type = LidarType::RSAIRY;
  }
  else if (lidar_type == "RS16")
  {
      param.lidar_type = LidarType::RS16;
  }
  else if (lidar_type == "RS32")
  {
      param.lidar_type = LidarType::RS32;
  }
  else if (lidar_type == "RSBP")
  {
      param.lidar_type = LidarType::RSBP;
  }
  else if (lidar_type == "RSAIRYLITE_ETH")
  {
      param.lidar_type = LidarType::RSAIRYLITE_ETH;
  }
  else if (lidar_type == "RSFAIRY")
  {
      param.lidar_type = LidarType::RSFAIRY;
  }
  else if (lidar_type == "RSHELIOS")
  {
      param.lidar_type = LidarType::RSHELIOS;
  }
  else if (lidar_type == "RSHELIOS_16P")
  {
      param.lidar_type = LidarType::RSHELIOS_16P;
  }
  else if (lidar_type == "RS128")
  {
      param.lidar_type = LidarType::RS128;
  }
  else if (lidar_type == "RSP80")
  {
      param.lidar_type = LidarType::RSP80;
  }
  else
  {
      throw std::runtime_error(
          "Unsupported LiDAR type: " + lidar_type
      );
  }
  // Add these LiDAR types if needed RSP48,RSM1, RSM1_JUMBO, RSM2, RSM3, RSE1, RSMX, RSEMX.

  param.input_type = InputType::ONLINE_LIDAR;
  param.input_param.host_address = config.host_ip;
  param.input_param.msop_port = config.msop_port;   ///< Set the lidar msop port number, the default is 6699
  param.input_param.difop_port = config.difop_port;  ///< Set the lidar difop port number, the default is 7788
#if ENABLE_IMU_PARSE
  param.input_param.imu_port = config.imu_port;   ///< Set the lidar imu port number, the default is 0
#endif
  param.input_param.user_layer_bytes = config.user_layer_bytes;
  param.input_param.tail_layer_bytes = config.tail_layer_bytes;
  param.decoder_param.min_distance = config.min_distance;
  param.decoder_param.max_distance = config.max_distance;
  param.decoder_param.use_lidar_clock = config.use_lidar_clock;
  param.decoder_param.dense_points = config.dense_points;
  param.decoder_param.ts_first_point = config.ts_first_point;
  param.decoder_param.start_angle = config.start_angle;
  param.decoder_param.end_angle = config.end_angle;

  gyro_bias.x = config.calibration_x;
  gyro_bias.y = config.calibration_y;
  gyro_bias.z = config.calibration_z;

  return param;
}


int main(int argc, char* argv[])
{
  RS_TITLE << "------------------------------------------------------" << RS_REND;
  RS_TITLE << "            RS_Driver Core Version: v" << getDriverVersion() << RS_REND;
  RS_TITLE << "------------------------------------------------------" << RS_REND;
  
  Config config = loadConfig("../config/config.yaml");
  RSDriverParam param = loadParams(config);
  param.print();
   
  LidarDriver<PointCloudMsg> driver;               ///< Declare the driver object
  driver.regPointCloudCallback(driverGetPointCloudFromCallerCallback, driverReturnPointCloudToCallerCallback); ///< Register the point cloud callback functions
  driver.regExceptionCallback(exceptionCallback);  ///< Register the exception callback function
#if ENABLE_IMU_PARSE
  driver.regImuDataCallback(driverGetIMUDataFromCallerCallback, driverReturnImuDataToCallerCallback);
#endif

  if (!driver.init(param))                         ///< Call the init function
  {
    RS_ERROR << "Driver Initialize Error..." << RS_REND;
    return -1;
  }

  std::thread cloud_handle_thread = std::thread(processCloud_range_reflective);
#if ENABLE_IMU_PARSE
  std::thread imuData_handle_thread = std::thread(processImuData);
#endif
  driver.start();  ///< The driver thread will start
  RS_DEBUG << "RoboSense Lidar-Driver Linux online demo start......" << RS_REND;

#ifdef ORDERLY_EXIT
  std::this_thread::sleep_for(std::chrono::seconds(10));

  driver.stop();

  to_exit_process = true;
  cloud_handle_thread.join();
#else
  while (true)
  {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
#endif

  return 0;
}
