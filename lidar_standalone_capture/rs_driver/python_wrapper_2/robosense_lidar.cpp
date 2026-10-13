#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <rs_driver/api/lidar_driver.hpp>
#include <rs_driver/msg/point_cloud_msg.hpp>

#include <yaml-cpp/yaml.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace py = pybind11;

using namespace robosense::lidar;


/* ============================================================
 * GLOBAL QUEUES
 * ============================================================ */

SyncQueue<std::shared_ptr<PointCloudT<PointXYZIRT>>> free_cloud_queue;
SyncQueue<std::shared_ptr<PointCloudT<PointXYZIRT>>> stuffed_cloud_queue;

SyncQueue<std::shared_ptr<ImuData>> free_imu_queue;
SyncQueue<std::shared_ptr<ImuData>> stuffed_imu_queue;

// Set by the driver when a PCAP file has been read to the end.
std::atomic<bool> pcap_finished{false};

// Columns per point: x, y, z, range, intensity, time offset (s) from the
// cloud timestamp. The offset, not the absolute time, is stored because
// float32 cannot hold LiDAR-clock seconds to microseconds.
constexpr size_t COLS = 6;


/* ============================================================
 * IMU STRUCTURE
 * ============================================================ */

struct IMUState
{
    double timestamp = 0.0;

    double angular_velocity_x = 0.0;
    double angular_velocity_y = 0.0;
    double angular_velocity_z = 0.0;

    double linear_acceleration_x = 0.0;
    double linear_acceleration_y = 0.0;
    double linear_acceleration_z = 0.0;

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double velocity_z = 0.0;

    double speed = 0.0;

    bool state = false;
};


/* ============================================================
 * LIDAR CLASS
 * ============================================================ */

class RoboSenseLidar
{
private:

    LidarDriver<PointCloudT<PointXYZIRT>> driver;

    std::atomic<bool> running{false};

    std::thread cloud_thread;
    std::thread imu_thread;

    std::mutex cloud_mutex;
    std::mutex imu_mutex;

    py::object cloud_callback = py::none();
    py::object imu_callback = py::none();

    std::vector<float> latest_cloud;

    // Seconds, as in PointCloudT::timestamp; an integer type truncated
    // this to whole seconds, so ~10 frames shared one timestamp.
    double latest_cloud_timestamp = 0.0;
    uint32_t latest_cloud_seq = 0;

    IMUState latest_imu;

    bool remove_nan = true;

    /* IMU state */

    bool imu_initialized = false;

    double previous_imu_timestamp = 0.0;

    double roll = 0.0;
    double pitch = 0.0;
    double yaw = 0.0;

    double velocity_x = 0.0;
    double velocity_y = 0.0;
    double velocity_z = 0.0;

    double gyro_bias_x = 0.0;
    double gyro_bias_y = 0.0;
    double gyro_bias_z = 0.0;

    double accel_filter_alpha = 0.98;


public:

    RoboSenseLidar()
    {
    }


    ~RoboSenseLidar()
    {
        stop();
    }


    /* ========================================================
     * POINT CLOUD CALLBACK
     * ======================================================== */

    static std::shared_ptr<PointCloudT<PointXYZIRT>>
    getPointCloud()
    {
        auto msg = free_cloud_queue.pop();

        if (msg)
            return msg;

        return std::make_shared<PointCloudT<PointXYZIRT>>();
    }


    static void
    returnPointCloud(
        std::shared_ptr<PointCloudT<PointXYZIRT>> msg)
    {
        stuffed_cloud_queue.push(msg);
    }


    /* ========================================================
     * IMU CALLBACK
     * ======================================================== */

    static std::shared_ptr<ImuData>
    getIMU()
    {
        auto msg = free_imu_queue.pop();

        if (msg)
            return msg;

        return std::make_shared<ImuData>();
    }


    static void
    returnIMU(
        const std::shared_ptr<ImuData>& msg)
    {
        stuffed_imu_queue.push(msg);
    }


    /* ========================================================
     * EXCEPTION CALLBACK
     * ======================================================== */

    static void
    exceptionCallback(const Error& error)
    {
        if (error.error_code == ERRCODE_PCAPEXIT)
        {
            pcap_finished = true;
            return;
        }

        RS_WARNING << error.toString() << RS_REND;
    }


    /* ========================================================
     * START
     * ======================================================== */

    /*
     * pcap_path empty: receive live packets from the LiDAR.
     * pcap_path set:   decode a recorded PCAP file once, paced at pcap_rate
     *                  times real time; input_finished() reports its end.
     */
    void start(const std::string& config_file,
               const std::string& pcap_path = "",
               float pcap_rate = 1.0f)
    {
        if (running)
            return;

        YAML::Node config = YAML::LoadFile(config_file);

        RSDriverParam param;

        std::string lidar_type =
            config["lidar"]["lidar_type"].as<std::string>();


        if (lidar_type == "RSFAIRY")
        {
            param.lidar_type = LidarType::RSFAIRY;
        }
        else if (lidar_type == "RSAIRY")
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
        else
        {
            throw std::runtime_error(
                "Unsupported LiDAR type: " + lidar_type);
        }


        /* ---------------------------------------------
         * INPUT
         * --------------------------------------------- */

        param.input_type =
            pcap_path.empty() ? InputType::ONLINE_LIDAR
                              : InputType::PCAP_FILE;

        param.input_param.pcap_path =
            pcap_path;

        param.input_param.pcap_repeat =
            false;

        param.input_param.pcap_rate =
            pcap_rate;

        pcap_finished =
            false;

        param.input_param.host_address =
            config["network"]["host_ip"].as<std::string>();

        param.input_param.msop_port =
            config["lidar"]["msop_port"].as<int>();

        param.input_param.difop_port =
            config["lidar"]["difop_port"].as<int>();

        param.input_param.imu_port =
            config["lidar"]["imu_port"].as<int>();

        param.input_param.user_layer_bytes =
            config["lidar"]["user_layer_bytes"].as<int>();

        param.input_param.tail_layer_bytes =
            config["lidar"]["tail_layer_bytes"].as<int>();


        /* ---------------------------------------------
         * DECODER
         * --------------------------------------------- */

        param.decoder_param.min_distance =
            config["lidar"]["min_distance"].as<float>();

        param.decoder_param.max_distance =
            config["lidar"]["max_distance"].as<float>();

        param.decoder_param.use_lidar_clock =
            config["lidar"]["use_lidar_clock"].as<bool>();

        param.decoder_param.dense_points =
            config["lidar"]["dense_points"].as<bool>();

        param.decoder_param.ts_first_point =
            config["lidar"]["ts_first_point"].as<bool>();

        param.decoder_param.start_angle =
            config["lidar"]["start_angle"].as<float>();

        param.decoder_param.end_angle =
            config["lidar"]["end_angle"].as<float>();

        // Where one sweep ends and the next begins. rs_driver's default
        // (0 deg) puts the boundary straight ahead, splitting the forward
        // field of view across two sweeps 100 ms apart. Optional key.
        if (config["lidar"]["split_angle"])
        {
            param.decoder_param.split_angle =
                config["lidar"]["split_angle"].as<float>();
        }


        /* ---------------------------------------------
         * CALLBACKS
         * --------------------------------------------- */

        driver.regPointCloudCallback(
            getPointCloud,
            returnPointCloud
        );

        driver.regImuDataCallback(
            getIMU,
            returnIMU
        );

        driver.regExceptionCallback(
            exceptionCallback
        );


        /* ---------------------------------------------
         * INITIALIZE
         * --------------------------------------------- */

        if (!driver.init(param))
        {
            throw std::runtime_error(
                "RoboSense driver initialization failed");
        }


        running = true;


        cloud_thread =
            std::thread(
                &RoboSenseLidar::processCloud,
                this
            );


        imu_thread =
            std::thread(
                &RoboSenseLidar::processIMU,
                this
            );


        driver.start();
    }


    /* ========================================================
     * POINT CLOUD PROCESSING
     * ======================================================== */

    void processCloud()
    {
        while (running)
        {
            auto msg =
                stuffed_cloud_queue.popWait();

            if (!msg)
                continue;

            // stop() wakes this thread with a dummy message; no callback
            // may run after that, as stop() may be joining us.
            if (!running)
                break;


            std::vector<float> cloud;


            /*
             * Each point:
             *
             * x
             * y
             * z
             * range
             * intensity
             * time offset from msg->timestamp (s), per point
             */

            cloud.reserve(msg->points.size() * COLS);


            for (const auto& p : msg->points)
            {
                const float x = p.x;
                const float y = p.y;
                const float z = p.z;


                /* -----------------------------------------
                 * Remove NaN / Inf
                 * ----------------------------------------- */

                if (remove_nan)
                {
                    if (!std::isfinite(x) ||
                        !std::isfinite(y) ||
                        !std::isfinite(z))
                    {
                        continue;
                    }
                }


                /* -----------------------------------------
                 * Calculate range
                 * ----------------------------------------- */

                const float range =
                    std::sqrt(
                        x * x +
                        y * y +
                        z * z
                    );


                if (!std::isfinite(range))
                    continue;


                /* -----------------------------------------
                 * Intensity / reflectivity
                 * ----------------------------------------- */

                const float intensity =
                    static_cast<float>(p.intensity);


                /* -----------------------------------------
                 * Store point
                 * ----------------------------------------- */

                cloud.push_back(x);
                cloud.push_back(y);
                cloud.push_back(z);
                cloud.push_back(range);
                cloud.push_back(intensity);
                cloud.push_back(static_cast<float>(p.timestamp - msg->timestamp));
            }


            /* ---------------------------------------------
             * Update latest cloud
             * --------------------------------------------- */

            double cloud_timestamp;
            uint32_t cloud_seq;

            {
                std::lock_guard<std::mutex> lock(
                    cloud_mutex
                );

                latest_cloud =
                    std::move(cloud);

                latest_cloud_timestamp =
                    msg->timestamp;

                latest_cloud_seq =
                    msg->seq;

                cloud_timestamp =
                    latest_cloud_timestamp;

                cloud_seq =
                    latest_cloud_seq;
            }


            /* ---------------------------------------------
             * Python callback
             * --------------------------------------------- */

            {
                py::gil_scoped_acquire gil;

                if (!cloud_callback.is_none())
                {
                    try
                    {
                        /*
                         * pybind11 3.0.4 compatible ShapeContainer.
                         *
                         * DO NOT use:
                         *
                         * shape.push_back(...)
                         *
                         * ShapeContainer is an any_container.
                         */

                        py::array::ShapeContainer shape{
                            static_cast<py::ssize_t>(
                                latest_cloud.size() / COLS
                            ),
                            static_cast<py::ssize_t>(COLS)
                        };


                        py::array_t<float> array(shape);


                        if (!latest_cloud.empty())
                        {
                            std::memcpy(
                                array.mutable_data(),
                                latest_cloud.data(),
                                latest_cloud.size() *
                                    sizeof(float)
                            );
                        }


                        cloud_callback(
                            array,
                            cloud_timestamp,
                            cloud_seq
                        );
                    }
                    catch (const py::error_already_set&)
                    {
                        PyErr_Print();
                    }
                }
            }


            /* ---------------------------------------------
             * Return message to free queue
             * --------------------------------------------- */

            free_cloud_queue.push(msg);
        }
    }


    /* ========================================================
     * IMU PROCESSING
     * ======================================================== */

    void processIMU()
    {
        while (running)
        {
            auto msg =
                stuffed_imu_queue.popWait();

            if (!msg)
                continue;

            if (!running)
                break;


            /* ---------------------------------------------
             * Timestamp
             * --------------------------------------------- */

            double timestamp =
                static_cast<double>(
                    msg->timestamp
                );


            /* ---------------------------------------------
             * Gyroscope
             * --------------------------------------------- */

            double gx =
                msg->angular_velocity_x -
                gyro_bias_x;

            double gy =
                msg->angular_velocity_y -
                gyro_bias_y;

            double gz =
                msg->angular_velocity_z -
                gyro_bias_z;


            /* ---------------------------------------------
             * Accelerometer
             * --------------------------------------------- */

            double ax =
                msg->linear_acceleration_x;

            double ay =
                msg->linear_acceleration_y;

            double az =
                msg->linear_acceleration_z;


            /* ---------------------------------------------
             * IMU initialization
             * --------------------------------------------- */

            if (!imu_initialized)
            {
                previous_imu_timestamp =
                    timestamp;


                roll =
                    std::atan2(
                        ay,
                        az
                    );


                pitch =
                    std::atan2(
                        -ax,
                        std::sqrt(
                            ay * ay +
                            az * az
                        )
                    );


                yaw = 0.0;


                velocity_x = 0.0;
                velocity_y = 0.0;
                velocity_z = 0.0;


                imu_initialized = true;
            }
            else
            {
                double dt =
                    timestamp -
                    previous_imu_timestamp;


                previous_imu_timestamp =
                    timestamp;


                if (dt > 0.0 && dt < 1.0)
                {
                    /* ---------------------------------
                     * Gyroscope integration
                     * --------------------------------- */

                    double roll_gyro =
                        roll + gx * dt;

                    double pitch_gyro =
                        pitch + gy * dt;

                    double yaw_gyro =
                        yaw + gz * dt;


                    /* ---------------------------------
                     * Accelerometer attitude
                     * --------------------------------- */

                    double roll_acc =
                        std::atan2(
                            ay,
                            az
                        );


                    double pitch_acc =
                        std::atan2(
                            -ax,
                            std::sqrt(
                                ay * ay +
                                az * az
                            )
                        );


                    /* ---------------------------------
                     * Complementary filter
                     * --------------------------------- */

                    roll =
                        accel_filter_alpha *
                            roll_gyro
                        +
                        (1.0 -
                         accel_filter_alpha) *
                            roll_acc;


                    pitch =
                        accel_filter_alpha *
                            pitch_gyro
                        +
                        (1.0 -
                         accel_filter_alpha) *
                            pitch_acc;


                    /*
                     * Accelerometer cannot correct yaw.
                     */

                    yaw =
                        yaw_gyro;


                    /* ---------------------------------
                     * Rotation matrix
                     * --------------------------------- */

                    double cr =
                        std::cos(roll);

                    double sr =
                        std::sin(roll);

                    double cp =
                        std::cos(pitch);

                    double sp =
                        std::sin(pitch);

                    double cy =
                        std::cos(yaw);

                    double sy =
                        std::sin(yaw);


                    double r11 =
                        cy * cp;


                    double r12 =
                        cy * sp * sr -
                        sy * cr;


                    double r13 =
                        cy * sp * cr +
                        sy * sr;


                    double r21 =
                        sy * cp;


                    double r22 =
                        sy * sp * sr +
                        cy * cr;


                    double r23 =
                        sy * sp * cr -
                        cy * sr;


                    double r31 =
                        -sp;


                    double r32 =
                        cp * sr;


                    double r33 =
                        cp * cr;


                    /* ---------------------------------
                     * Transform acceleration
                     * --------------------------------- */

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


                    /* ---------------------------------
                     * Remove gravity
                     * --------------------------------- */

                    world_az -=
                        9.80665;


                    /* ---------------------------------
                     * Integrate velocity
                     * --------------------------------- */

                    velocity_x +=
                        world_ax * dt;

                    velocity_y +=
                        world_ay * dt;

                    velocity_z +=
                        world_az * dt;
                }
            }


            /* ---------------------------------------------
             * Calculate speed
             * --------------------------------------------- */

            double speed =
                std::sqrt(
                    velocity_x * velocity_x +
                    velocity_y * velocity_y +
                    velocity_z * velocity_z
                );


            /* ---------------------------------------------
             * Build IMU state
             * --------------------------------------------- */

            IMUState imu;


            imu.timestamp =
                timestamp;


            imu.angular_velocity_x =
                msg->angular_velocity_x;

            imu.angular_velocity_y =
                msg->angular_velocity_y;

            imu.angular_velocity_z =
                msg->angular_velocity_z;


            imu.linear_acceleration_x =
                ax;

            imu.linear_acceleration_y =
                ay;

            imu.linear_acceleration_z =
                az;


            imu.roll =
                roll;

            imu.pitch =
                pitch;

            imu.yaw =
                yaw;


            imu.velocity_x =
                velocity_x;

            imu.velocity_y =
                velocity_y;

            imu.velocity_z =
                velocity_z;


            imu.speed =
                speed;


            imu.state =
                msg->state;


            /* ---------------------------------------------
             * Store latest IMU
             * --------------------------------------------- */

            {
                std::lock_guard<std::mutex> lock(
                    imu_mutex
                );

                latest_imu =
                    imu;
            }


            /* ---------------------------------------------
             * Python callback
             * --------------------------------------------- */

            {
                py::gil_scoped_acquire gil;

                if (!imu_callback.is_none())
                {
                    try
                    {
                        py::dict data;


                        data["timestamp"] =
                            imu.timestamp;


                        data["angular_velocity"] =
                            py::make_tuple(
                                imu.angular_velocity_x,
                                imu.angular_velocity_y,
                                imu.angular_velocity_z
                            );


                        data["linear_acceleration"] =
                            py::make_tuple(
                                imu.linear_acceleration_x,
                                imu.linear_acceleration_y,
                                imu.linear_acceleration_z
                            );


                        data["roll_rad"] =
                            imu.roll;


                        data["pitch_rad"] =
                            imu.pitch;


                        data["yaw_rad"] =
                            imu.yaw;


                        data["roll_deg"] =
                            imu.roll *
                            180.0 /
                            M_PI;


                        data["pitch_deg"] =
                            imu.pitch *
                            180.0 /
                            M_PI;


                        data["yaw_deg"] =
                            imu.yaw *
                            180.0 /
                            M_PI;


                        data["velocity"] =
                            py::make_tuple(
                                imu.velocity_x,
                                imu.velocity_y,
                                imu.velocity_z
                            );


                        data["speed_mps"] =
                            imu.speed;


                        data["state"] =
                            imu.state;


                        imu_callback(data);
                    }
                    catch (const py::error_already_set&)
                    {
                        PyErr_Print();
                    }
                }
            }


            /* ---------------------------------------------
             * Return IMU message
             * --------------------------------------------- */

            free_imu_queue.push(msg);
        }
    }


    /* ========================================================
     * GET POINT CLOUD
     * ======================================================== */

    py::dict getPointCloudPython()
    {
        std::lock_guard<std::mutex> lock(
            cloud_mutex
        );


        py::dict result;


        const size_t n =
            latest_cloud.size() / COLS;


        /*
         * pybind11 3.0.4 compatible.
         */

        py::array::ShapeContainer shape{
            static_cast<py::ssize_t>(n),
            static_cast<py::ssize_t>(COLS)
        };


        py::array_t<float> points(shape);


        if (!latest_cloud.empty())
        {
            std::memcpy(
                points.mutable_data(),
                latest_cloud.data(),
                latest_cloud.size() *
                    sizeof(float)
            );
        }


        result["points"] =
            points;


        result["timestamp"] =
            latest_cloud_timestamp;


        result["seq"] =
            latest_cloud_seq;


        return result;
    }


    /* ========================================================
     * GET IMU
     * ======================================================== */

    py::dict getIMUPython()
    {
        std::lock_guard<std::mutex> lock(
            imu_mutex
        );


        py::dict result;


        result["timestamp"] =
            latest_imu.timestamp;


        result["angular_velocity"] =
            py::make_tuple(
                latest_imu.angular_velocity_x,
                latest_imu.angular_velocity_y,
                latest_imu.angular_velocity_z
            );


        result["linear_acceleration"] =
            py::make_tuple(
                latest_imu.linear_acceleration_x,
                latest_imu.linear_acceleration_y,
                latest_imu.linear_acceleration_z
            );


        result["roll_rad"] =
            latest_imu.roll;


        result["pitch_rad"] =
            latest_imu.pitch;


        result["yaw_rad"] =
            latest_imu.yaw;


        result["roll_deg"] =
            latest_imu.roll *
            180.0 /
            M_PI;


        result["pitch_deg"] =
            latest_imu.pitch *
            180.0 /
            M_PI;


        result["yaw_deg"] =
            latest_imu.yaw *
            180.0 /
            M_PI;


        result["velocity"] =
            py::make_tuple(
                latest_imu.velocity_x,
                latest_imu.velocity_y,
                latest_imu.velocity_z
            );


        result["speed_mps"] =
            latest_imu.speed;


        result["state"] =
            latest_imu.state;


        return result;
    }


    /* ========================================================
     * CALLBACK REGISTRATION
     * ======================================================== */

    void setPointCloudCallback(
        py::object callback)
    {
        py::gil_scoped_acquire gil;

        cloud_callback =
            callback;
    }


    void setIMUCallback(
        py::object callback)
    {
        py::gil_scoped_acquire gil;

        imu_callback =
            callback;
    }


    /* ========================================================
     * PARAMETERS
     * ======================================================== */

    void setRemoveNaN(
        bool enable)
    {
        remove_nan =
            enable;
    }


    void setGyroBias(
        double x,
        double y,
        double z)
    {
        gyro_bias_x =
            x;

        gyro_bias_y =
            y;

        gyro_bias_z =
            z;
    }


    /* ========================================================
     * RESET IMU
     * ======================================================== */

    void resetIMU()
    {
        std::lock_guard<std::mutex> lock(
            imu_mutex
        );


        imu_initialized =
            false;


        previous_imu_timestamp =
            0.0;


        roll =
            0.0;

        pitch =
            0.0;

        yaw =
            0.0;


        velocity_x =
            0.0;

        velocity_y =
            0.0;

        velocity_z =
            0.0;
    }


    /* ========================================================
     * STOP
     * ======================================================== */

    void stop()
    {
        if (!running)
            return;


        running =
            false;


        /*
         * Stop the driver's receive/decode threads first, so no new
         * messages are produced while the processing threads exit.
         */

        driver.stop();


        /*
         * The processing threads may be waiting inside
         * popWait(). Push dummy messages to wake them up.
         */


        stuffed_cloud_queue.push(
            std::make_shared<PointCloudT<PointXYZIRT>>()
        );


        stuffed_imu_queue.push(
            std::make_shared<ImuData>()
        );


        /*
         * The processing threads acquire the GIL for Python callbacks.
         * Joining them while holding the GIL deadlocks, so release it
         * when held (Python calls and object destruction both hold it).
         */

        std::unique_ptr<py::gil_scoped_release> release;

        if (PyGILState_Check())
            release.reset(new py::gil_scoped_release());


        if (cloud_thread.joinable())
            cloud_thread.join();


        if (imu_thread.joinable())
            imu_thread.join();
    }


    /* ========================================================
     * RUNNING STATUS
     * ======================================================== */

    bool isRunning() const
    {
        return running;
    }


    bool inputFinished() const
    {
        return pcap_finished;
    }
};


/* ============================================================
 * PYBIND11 MODULE
 * ============================================================ */

PYBIND11_MODULE(
    robosense_lidar,
    m
)
{
    m.doc() =
        "RoboSense LiDAR Python wrapper";


    py::class_<RoboSenseLidar>(
        m,
        "Lidar"
    )


    .def(
        py::init<>()
    )


    .def(
        "start",
        &RoboSenseLidar::start,
        py::arg("config_file"),
        py::arg("pcap_path") = "",
        py::arg("pcap_rate") = 1.0f,
        // start() touches no Python objects. Releasing the GIL lets it block
        // (e.g. opening a FIFO that a Python thread is about to feed)
        // without deadlocking that thread.
        py::call_guard<py::gil_scoped_release>()
    )


    .def(
        "stop",
        &RoboSenseLidar::stop
    )


    .def(
        "is_running",
        &RoboSenseLidar::isRunning
    )


    .def(
        "input_finished",
        &RoboSenseLidar::inputFinished
    )


    .def(
        "get_pointcloud",
        &RoboSenseLidar::getPointCloudPython
    )


    .def(
        "get_imu",
        &RoboSenseLidar::getIMUPython
    )


    .def(
        "set_pointcloud_callback",
        &RoboSenseLidar::setPointCloudCallback
    )


    .def(
        "set_imu_callback",
        &RoboSenseLidar::setIMUCallback
    )


    .def(
        "set_remove_nan",
        &RoboSenseLidar::setRemoveNaN
    )


    .def(
        "set_gyro_bias",
        &RoboSenseLidar::setGyroBias
    )


    .def(
        "reset_imu",
        &RoboSenseLidar::resetIMU
    );
}