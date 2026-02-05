/*
 * File: dtg_recorder_node.cpp
 * ---------------------------
 * Standalone recorder node for MultiDTG:
 * - Calls the `dtg_snapshot` Trigger service periodically
 * - Writes the returned CSV string to disk
 */

#include <ros/ros.h>
#include <std_srvs/Trigger.h>

#include <fstream>
#include <iomanip>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <vector>

class DTGRecorderNode {
public:
  DTGRecorderNode() : nh_(), pnh_("~") {
    pnh_.param<std::string>("service_name", serviceName_, "dtg_snapshot");
    pnh_.param<std::string>("robot_namespace_prefix", robotNamespacePrefix_, "/uav");
    pnh_.param<int>("robot_id", robotId_, 0);
    pnh_.param<std::string>("output_directory", outputDir_, "/home/cerlab/ros1/logs/dtg_dumps");
    pnh_.param<std::string>("file_prefix", filePrefix_, "dtg_snapshot");
    pnh_.param<double>("interval", intervalSec_, 10.0);

    if (intervalSec_ <= 0.0) {
      ROS_WARN("[dtg_recorder_node] interval <= 0, forcing to 10.0s");
      intervalSec_ = 10.0;
    }

    if (!ensureDirectory(outputDir_)) {
      throw std::runtime_error("Failed to create output directory");
    }

    const std::string srvName = serviceForRobot(robotId_);
    client_ = nh_.serviceClient<std_srvs::Trigger>(srvName);
    timer_ = nh_.createTimer(ros::Duration(intervalSec_), &DTGRecorderNode::timerCB, this);

    ROS_INFO("[dtg_recorder_node] Recording enabled. robot_id=%d service=%s interval=%.2f dir=%s prefix=%s",
             robotId_, srvName.c_str(), intervalSec_, outputDir_.c_str(), filePrefix_.c_str());
  }

private:
  static bool ensureDirectory(const std::string& dir) {
    if (dir.empty()) return false;
    std::string path;
    path.reserve(dir.size());

    for (size_t i = 0; i < dir.size(); ++i) {
      const char c = dir[i];
      path.push_back(c);
      if (c == '/' || i == dir.size() - 1) {
        if (path.size() == 1 && path[0] == '/') continue;
        if (::mkdir(path.c_str(), 0755) != 0) {
          if (errno == EEXIST) continue;
          ROS_ERROR("[dtg_recorder_node] mkdir('%s') failed: %s", path.c_str(), std::strerror(errno));
          return false;
        }
      }
    }
    return true;
  }

  std::string serviceForRobot(int robotId) const {
    // If serviceName is absolute, use it directly.
    if (!serviceName_.empty() && serviceName_.front() == '/') return serviceName_;
    
    // Example: /uav0/murder_node/dtg_snapshot
    std::string srv = robotNamespacePrefix_ + std::to_string(robotId) + "/" + serviceName_;
    return srv;
  }

  void timerCB(const ros::TimerEvent&) {
    std_srvs::Trigger srv;
    if (!client_.call(srv)) {
      ROS_WARN("[dtg_recorder_node] Service call failed");
      return;
    }
    if (!srv.response.success) {
      ROS_WARN("[dtg_recorder_node] Service returned success=false: %s", srv.response.message.c_str());
      return;
    }

    const uint64_t stampNs = static_cast<uint64_t>(ros::Time::now().toNSec());
    const std::string filename =
        filePrefix_ + std::string("_rid") + std::to_string(robotId_) +
        std::string("_") + std::to_string(stampNs) + ".csv";
    const std::string outPath = outputDir_ + (outputDir_.back() == '/' ? "" : "/") + filename;

    std::ofstream ofs(outPath, std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) {
      ROS_WARN("[dtg_recorder_node] Failed to open '%s' for writing", outPath.c_str());
      return;
    }
    ofs << "# RECORDER\n";
    ofs << "robot_id," << robotId_ << "\n";
    ofs << srv.response.message;
    ofs.close();

    if (!ofs) {
      ROS_WARN("[dtg_recorder_node] Failed to write '%s'", outPath.c_str());
      return;
    }

    ROS_INFO("[dtg_recorder_node] Wrote snapshot: %s", outPath.c_str());
  }

  ros::NodeHandle nh_;
  ros::NodeHandle pnh_;
  ros::ServiceClient client_;
  ros::Timer timer_;

  std::string serviceName_;
  std::string robotNamespacePrefix_;
  int robotId_{0};
  std::string outputDir_;
  std::string filePrefix_;
  double intervalSec_{10.0};
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "dtg_recorder_node");
  try {
    DTGRecorderNode node;
    ros::spin();
  } catch (...) {
    return 1;
  }
  return 0;
}
