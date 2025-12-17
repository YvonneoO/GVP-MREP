#include <ros/ros.h>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Geometry>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <swarm_exp_msgs/LocalTraj.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <algorithm>
#include <cmath>
#include <deque>
#include <list>
#include <string>

ros::Subscriber traj_sub_, odom_sub_, sparse_waypoints_sub_;
ros::Publisher cmd_pub_;
ros::Timer control_timer_;

std::list<swarm_exp_msgs::LocalTraj> traj_queue_;
std::deque<geometry_msgs::PoseStamped> path_queue_;
geometry_msgs::PoseStamped rotation_target_;

Eigen::Vector3d recover_pt_;
Eigen::Vector3d robot_pos_;
nav_msgs::Odometry latest_odom_;

size_t target_idx_;
double robot_yaw_;
double integral_a_;
double error_a_last_;
double sample_dt_;
double desired_vel_;
double desired_ang_vel_;
double reach_goal_distance_;
double vel_lower_bound_;
double ang_vel_lower_bound_;
double Kp_a_;
double Ki_a_;
double Kd_a_;
double Kp_d_;
double Ki_d_;
double Kd_d_;
double yaw_goal_tolerance_;
double control_rate_;
double initial_backward_distance_;
double rotation_target_pos_thresh_;
double rotation_target_yaw_thresh_;
double goal_update_pos_thresh_;
double goal_update_yaw_thresh_;

int traj_state_;
bool path_active_;
bool rotation_mode_;
bool have_odom_;
bool initial_backward_done_;
bool have_rotation_target_;
bool have_goal_;
Eigen::Vector3d initial_odom_pos_;

geometry_msgs::Twist zero_cmd_;
std::string traj_topic_;
std::string odom_topic_;
std::string cmd_topic_;
std::string sparse_waypoints_topic_;
geometry_msgs::PoseStamped last_accepted_goal_;

geometry_msgs::Quaternion YawToQuat(double yaw) {
    tf2::Quaternion q;
    q.setRPY(0.0, 0.0, yaw);
    q.normalize();
    return tf2::toMsg(q);
}

double WrapAngle(double angle) {
    while (angle > M_PI) angle -= 2.0 * M_PI;
    while (angle < -M_PI) angle += 2.0 * M_PI;
    return angle;
}

void ResetHeadingController() {
    integral_a_ = 0.0;
    error_a_last_ = 0.0;
}

void PublishStop() {
    cmd_pub_.publish(zero_cmd_);
}

void BuildRecoverPath() {
    path_queue_.clear();
    geometry_msgs::PoseStamped pose;
    pose.header.frame_id = "world";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = recover_pt_.x();
    pose.pose.position.y = recover_pt_.y();
    pose.pose.position.z = recover_pt_.z();
    pose.pose.orientation = YawToQuat(robot_yaw_);
    path_queue_.push_back(pose);
    ROS_WARN("publishing recover path for ugv");
    target_idx_ = 0;
    path_active_ = true;
    rotation_mode_ = false;
}

bool TryUpdateTraj() {
    if (traj_queue_.empty()) return false;

    const auto& msg = traj_queue_.front();
    if (msg.state == 1) {
        traj_state_ = 1;
        recover_pt_(0) = msg.recover_pt.x;
        recover_pt_(1) = msg.recover_pt.y;
        recover_pt_(2) = msg.recover_pt.z;
        BuildRecoverPath();
    } else {
        // This case should not be reached if TrajCallback filters messages correctly.
        traj_state_ = -1;
        path_active_ = false;
    }
    traj_queue_.pop_front();
    ResetHeadingController();
    return true;
}

void OdomCallback(const nav_msgs::OdometryConstPtr& odom) {
    latest_odom_ = *odom;
    bool was_first_odom = !have_odom_;
    have_odom_ = true;
    robot_pos_.x() = odom->pose.pose.position.x;
    robot_pos_.y() = odom->pose.pose.position.y;
    robot_pos_.z() = odom->pose.pose.position.z;
    Eigen::Quaterniond q(odom->pose.pose.orientation.w,
                         odom->pose.pose.orientation.x,
                         odom->pose.pose.orientation.y,
                         odom->pose.pose.orientation.z);
    robot_yaw_ = std::atan2(q.toRotationMatrix()(1, 0), q.toRotationMatrix()(0, 0));
    
    // Record initial position when odom is first received
    if (was_first_odom) {
        initial_odom_pos_ = robot_pos_;
        if (initial_backward_distance_ > 1e-3) {
            initial_backward_done_ = false;
            ROS_WARN("[INITIAL_BACKWARD] First odom received. Starting initial backward movement from [%.2f, %.2f, %.2f]", 
                    initial_odom_pos_.x(), initial_odom_pos_.y(), initial_odom_pos_.z());
        } else {
            initial_backward_done_ = true;
        }
    }
}

bool IsWaypointReached(const geometry_msgs::PoseStamped& target) {
    Eigen::Vector2d delta(target.pose.position.x - robot_pos_.x(),
                          target.pose.position.y - robot_pos_.y());
    return delta.norm() <= reach_goal_distance_;
}

bool ShouldAlignYaw(const geometry_msgs::PoseStamped& target) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(target.pose.orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    double yaw_error = WrapAngle(yaw - robot_yaw_);
    return std::abs(yaw_error) > yaw_goal_tolerance_;
}

bool IsYawAligned(const geometry_msgs::PoseStamped& target) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(target.pose.orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    double yaw_error = WrapAngle(yaw - robot_yaw_);
    return std::abs(yaw_error) <= yaw_goal_tolerance_;
}

double PoseYaw(const geometry_msgs::PoseStamped& pose) {
    tf2::Quaternion tf_q;
    tf2::fromMsg(pose.pose.orientation, tf_q);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
    return yaw;
}

bool RotationTargetChanged(const geometry_msgs::PoseStamped& current_target,
                           const geometry_msgs::PoseStamped& new_target) {
    Eigen::Vector2d delta(new_target.pose.position.x - current_target.pose.position.x,
                          new_target.pose.position.y - current_target.pose.position.y);
    double yaw_delta = WrapAngle(PoseYaw(new_target) - PoseYaw(current_target));
    return delta.norm() > rotation_target_pos_thresh_ ||
           std::abs(yaw_delta) > rotation_target_yaw_thresh_;
}

bool GoalChanged(const geometry_msgs::PoseStamped& current_goal,
                              const geometry_msgs::PoseStamped& new_goal) {
    Eigen::Vector2d delta(new_goal.pose.position.x - current_goal.pose.position.x,
                          new_goal.pose.position.y - current_goal.pose.position.y);
    double yaw_delta = WrapAngle(PoseYaw(new_goal) - PoseYaw(current_goal));
    return delta.norm() > goal_update_pos_thresh_ ||
           std::abs(yaw_delta) > goal_update_yaw_thresh_;
}

void GoalDelta(const geometry_msgs::PoseStamped& a,
               const geometry_msgs::PoseStamped& b,
               double& pos_delta,
               double& yaw_delta) {
    Eigen::Vector2d delta(b.pose.position.x - a.pose.position.x,
                          b.pose.position.y - a.pose.position.y);
    pos_delta = delta.norm();
    yaw_delta = WrapAngle(PoseYaw(b) - PoseYaw(a));
}

geometry_msgs::PoseStamped CreateRotationTarget(const geometry_msgs::PoseStamped& waypoint) {
    geometry_msgs::PoseStamped rot_target = waypoint;

    auto wrap = [](double a){
        while (a >  M_PI) a -= 2.0*M_PI;
        while (a < -M_PI) a += 2.0*M_PI;
        return a;
    };

    // yaw "forward" toward the waypoint
    const double dx = waypoint.pose.position.x - robot_pos_.x();
    const double dy = waypoint.pose.position.y - robot_pos_.y();
    const double yaw_fwd  = std::atan2(dy, dx);
    const double yaw_back = wrap(yaw_fwd + M_PI);

    // pick the nearer orientation to current yaw
    const double err_fwd  = std::abs(wrap(yaw_fwd  - robot_yaw_));
    const double err_back = std::abs(wrap(yaw_back - robot_yaw_));

    const double chosen = (err_back < err_fwd) ? yaw_back : yaw_fwd;
    rot_target.pose.orientation = YawToQuat(chosen);
    return rot_target;
}

void SparseWaypointsCallback(const nav_msgs::PathConstPtr& msg) {
    if (msg->poses.empty()) {
        path_active_ = false;
        ROS_WARN("Received empty sparse waypoint path.");
        return;
    }

    const geometry_msgs::PoseStamped& new_goal = msg->poses.back();
    bool have_current_path_goal = path_active_ && !path_queue_.empty();
    geometry_msgs::PoseStamped current_path_goal;
    if (have_current_path_goal) current_path_goal = path_queue_.back();

    bool keep_current_path = false;
    double pos_delta_last = 0.0, yaw_delta_last = 0.0;
    double pos_delta_curr = 0.0, yaw_delta_curr = 0.0;
    if (have_goal_) {
        GoalDelta(last_accepted_goal_, new_goal, pos_delta_last, yaw_delta_last);
    }
    if (have_current_path_goal) {
        GoalDelta(current_path_goal, new_goal, pos_delta_curr, yaw_delta_curr);
    }
    bool small_vs_last = have_goal_ && !GoalChanged(last_accepted_goal_, new_goal);
    bool small_vs_current = have_current_path_goal && !GoalChanged(current_path_goal, new_goal);
    if (small_vs_last || small_vs_current) {
        ROS_INFO("New sparse path final goal change below threshold (pos<=%.2f, yaw<=%.2f rad). Keeping current plan. Δpos_last=%.3f Δyaw_last=%.3f, Δpos_current=%.3f Δyaw_current=%.3f",
                 goal_update_pos_thresh_, goal_update_yaw_thresh_,
                 pos_delta_last, yaw_delta_last, pos_delta_curr, yaw_delta_curr);
        keep_current_path = true;
    }

    if (keep_current_path) return;

    path_queue_.clear();
    for (const auto& pose : msg->poses) {
        path_queue_.push_back(pose);
    }

    target_idx_ = 0;
    path_active_ = true;
    traj_state_ = 0; // Using 0 for sparse path
    ResetHeadingController();
    last_accepted_goal_ = path_queue_.back();
    have_goal_ = true;
    
    // Enter rotation mode for the first waypoint
    if (!path_queue_.empty() && have_odom_) {
        geometry_msgs::PoseStamped desired_target = CreateRotationTarget(path_queue_[target_idx_]);
        bool need_update = !have_rotation_target_ || RotationTargetChanged(rotation_target_, desired_target);
        rotation_target_ = desired_target;
        have_rotation_target_ = true;
        if (need_update && ShouldAlignYaw(rotation_target_)) {
            rotation_mode_ = true;
            ResetHeadingController();
            ROS_INFO("Received sparse waypoint path (%zu points). Rotating toward first waypoint.", path_queue_.size());
        } else {
            rotation_mode_ = false;
            ROS_INFO("Received sparse waypoint path (%zu points). Orientation change below threshold.", path_queue_.size());
        }
    } else {
        rotation_mode_ = false;
        ROS_INFO("Received and activated sparse waypoint path with %zu points.", path_queue_.size());
    }
}

void TrajCallback(const swarm_exp_msgs::LocalTrajConstPtr& traj) {
    if (traj->state == 1) {
        traj_queue_.push_back(*traj);
    } else if (traj->state == 2) {
        ROS_INFO("Ignoring polynomial trajectory for UGV, using sparse waypoints instead.");
    }
}

geometry_msgs::Twist ComputeControl(const geometry_msgs::PoseStamped& target, bool rotation_only) {
    geometry_msgs::Twist cmd;
    Eigen::Vector3d target_global(target.pose.position.x,
                                  target.pose.position.y,
                                  target.pose.position.z);
    Eigen::Quaterniond quat(latest_odom_.pose.pose.orientation.w,
                            latest_odom_.pose.pose.orientation.x,
                            latest_odom_.pose.pose.orientation.y,
                            latest_odom_.pose.pose.orientation.z);
    Eigen::Matrix3d rot = quat.toRotationMatrix();
    // rot is the rotation from body frame to map frame
    // To transform from map to body: target_local = rot^T * (target_global - robot_pos)
    Eigen::Vector3d robot_pos(latest_odom_.pose.pose.position.x,
                              latest_odom_.pose.pose.position.y,
                              latest_odom_.pose.pose.position.z);
    Eigen::Vector3d target_local = rot.transpose() * (target_global - robot_pos);

    double target_angle = 0.0;
    double current_angle = 0.0;
    if (rotation_only) {
        current_angle = robot_yaw_;
        tf2::Quaternion tf_q;
        tf2::fromMsg(target.pose.orientation, tf_q);
        double roll, pitch, yaw;
        tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);
        target_angle = yaw;
    } else {
        target_angle = std::atan2(target_local(1), target_local(0));
        if (std::abs(target_local(0)) < 1e-3) {
            target_angle = (target_local(1) >= 0.0) ? M_PI / 2.0 : -M_PI / 2.0;
        }
        if (target_angle > M_PI / 2.0) target_angle -= M_PI;
        else if (target_angle < -M_PI / 2.0) target_angle += M_PI;
    }

    double error_angle = WrapAngle(target_angle - current_angle);
    if (!rotation_only) {
        if (error_angle > M_PI / 2.0) error_angle -= M_PI;
        else if (error_angle < -M_PI / 2.0) error_angle += M_PI;
    }

    integral_a_ += error_angle;
    double derivative = error_angle - error_a_last_;
    error_a_last_ = error_angle;

    double output_angle_vel = Kp_a_ * error_angle + Ki_a_ * integral_a_ + Kd_a_ * derivative;
    if (rotation_only) {
        if (std::abs(output_angle_vel) >= desired_ang_vel_) {
            output_angle_vel = std::copysign(desired_ang_vel_, output_angle_vel);
        } else if (std::abs(output_angle_vel) < ang_vel_lower_bound_) {
            output_angle_vel = std::copysign(ang_vel_lower_bound_, output_angle_vel);
        }
        cmd.angular.z = output_angle_vel;
        cmd.linear.x = 0.0;
        return cmd;
    }

    double error_dis = std::hypot(target_local(0), target_local(1));
    double forward_sign = (target_local(0) >= 0.0) ? 1.0 : -1.0;
    if (std::abs(target_local(0)) < 1e-3) {
        forward_sign = (target_local(1) >= 0.0) ? 1.0 : -1.0;
    }
    double output_linear_vel = 0.0;
    if (error_dis > reach_goal_distance_) {
        output_linear_vel = Kp_d_ * error_dis * forward_sign;
        if (std::abs(output_linear_vel) >= desired_vel_) {
            output_linear_vel = std::copysign(desired_vel_, output_linear_vel);
        } else if (std::abs(output_linear_vel) < vel_lower_bound_) {
            output_linear_vel = std::copysign(vel_lower_bound_, output_linear_vel);
        }
    }

    cmd.angular.z = output_angle_vel;
    cmd.linear.x = output_linear_vel;
    return cmd;
}

void ControlLoop(const ros::TimerEvent&) {
    while (TryUpdateTraj()) {
    }
    if (!have_odom_) return;

    bool waypoint_command_ready = path_active_ && !path_queue_.empty();

    // initial backward movement
    if (initial_backward_distance_ > 1e-3 && !initial_backward_done_) {
        if (waypoint_command_ready) {
            initial_backward_done_ = true;
            PublishStop();
        } else {
            Eigen::Vector3d delta = robot_pos_ - initial_odom_pos_;
            double distance_moved = delta.norm();

            if (distance_moved >= initial_backward_distance_) {
                initial_backward_done_ = true;
                ROS_WARN("[INITIAL_BACKWARD] Completed initial backward movement. Distance moved: %.2f m. Proceeding with normal waypoint following.", 
                         distance_moved);
                PublishStop();
                return;
            }

            geometry_msgs::Twist cmd;
            cmd.linear.x = -desired_vel_;
            cmd.linear.y = 0.0;
            cmd.linear.z = 0.0;
            cmd.angular.x = 0.0;
            cmd.angular.y = 0.0;
            cmd.angular.z = 0.0;

            ROS_INFO("[INITIAL_BACKWARD] Moving backwards. Distance moved: %.2f/%.2f m, cmd.linear.x: %.2f", 
                     distance_moved, initial_backward_distance_, cmd.linear.x);
            cmd_pub_.publish(cmd);
            return;
        }
    }

    if (rotation_mode_) {
        geometry_msgs::Twist cmd = ComputeControl(rotation_target_, true);
        // Ensure rotation mode only outputs angular velocity, no linear velocity
        cmd.linear.x = 0.0;
        cmd.linear.y = 0.0;
        cmd.linear.z = 0.0;
        // ROS_INFO("ROTATION_MODE: angular.z: %f, linear.x: %f", cmd.angular.z, cmd.linear.x);
        cmd_pub_.publish(cmd);
        
        if (IsYawAligned(rotation_target_)) {
            rotation_mode_ = false;
            // ROS_INFO("Yaw aligned, exiting rotation_mode_, publishing stop");
            PublishStop();
        }
        return;
    }

    if (!path_active_ || target_idx_ >= path_queue_.size()) {
        // ROS_INFO("no more targets, publish stop cmd");
        PublishStop();
        return;
    }

    // If we have a path but not in rotation mode, check if we need to enter rotation mode
    // This handles the case where path was received before odom was available
    if (!rotation_mode_ && path_active_ && !path_queue_.empty()) {
        geometry_msgs::PoseStamped first_target = path_queue_[target_idx_];
        geometry_msgs::PoseStamped rot_target = CreateRotationTarget(first_target);
        bool need_update = !have_rotation_target_ || RotationTargetChanged(rotation_target_, rot_target);
        rotation_target_ = rot_target;
        have_rotation_target_ = true;
        if (need_update && ShouldAlignYaw(rotation_target_)) {
            rotation_mode_ = true;
            ResetHeadingController();
            // ROS_INFO("Entering rotation mode for waypoint %zu", target_idx_);
            return;
        }
    }

    geometry_msgs::PoseStamped target = path_queue_[target_idx_];

    if (target_idx_ >= path_queue_.size() - 1) {
        Eigen::Vector2d delta(target.pose.position.x - robot_pos_.x(),
                              target.pose.position.y - robot_pos_.y());
        // ROS_INFO("Approaching final waypoint. Distance: %f / %f", delta.norm(), reach_goal_distance_);
    }

    if (IsWaypointReached(target)) {
        if (target_idx_ >= path_queue_.size() - 1) {
            path_active_ = false;
            ROS_INFO("UGV reached final waypoint. Publishing stop command.");
            PublishStop(); // Ensure stop command is sent immediately upon path completion
            if (traj_state_ == 2 && ShouldAlignYaw(target)) {
                rotation_target_ = target;
                have_rotation_target_ = true;
                rotation_mode_ = true;
                ResetHeadingController();
                // PublishStop(); // This will be handled by rotation_mode_ logic
                return;
            }
            // If not entering rotation_mode, we already called PublishStop()
            return;
        }
        // Move to next waypoint and enter rotation mode if necessary
        ++target_idx_;
        geometry_msgs::PoseStamped next_target = path_queue_[target_idx_];
        geometry_msgs::PoseStamped desired_rotation = CreateRotationTarget(next_target);
        bool need_update = !have_rotation_target_ || RotationTargetChanged(rotation_target_, desired_rotation);
        rotation_target_ = desired_rotation;
        have_rotation_target_ = true;
        if (need_update && ShouldAlignYaw(rotation_target_)) {
            rotation_mode_ = true;
            ResetHeadingController();
            ROS_INFO("Reached waypoint %zu, rotating toward next waypoint (yaw: %f)", 
                     target_idx_ - 1, std::atan2(next_target.pose.position.y - robot_pos_.y(),
                                                next_target.pose.position.x - robot_pos_.x()));
        }
        return;
    }

    // Only translate if we're not in rotation mode
    if (!rotation_mode_) {
        geometry_msgs::Twist cmd = ComputeControl(target, false);
        // ROS_INFO("TRANSLATION_MODE: angular.z: %f, linear.x: %f, target_idx: %zu", 
        //          cmd.angular.z, cmd.linear.x, target_idx_);
        
        // Debug: check target_local to see why linear.x might be negative
        // Eigen::Vector3d target_global(target.pose.position.x,
        //                               target.pose.position.y,
        //                               target.pose.position.z);
        // Eigen::Quaterniond quat(latest_odom_.pose.pose.orientation.w,
        //                         latest_odom_.pose.pose.orientation.x,
        //                         latest_odom_.pose.pose.orientation.y,
        //                         latest_odom_.pose.pose.orientation.z);
        // Eigen::Matrix3d rot = quat.toRotationMatrix();
        // Eigen::Vector3d robot_pos(latest_odom_.pose.pose.position.x,
        //                           latest_odom_.pose.pose.position.y,
        //                           latest_odom_.pose.pose.position.z);
        // Eigen::Vector3d target_local = rot.transpose() * (target_global - robot_pos);
        // ROS_INFO("  target_local: [%f, %f, %f], robot_yaw: %f", 
        //          target_local(0), target_local(1), target_local(2), robot_yaw_);
        
        cmd_pub_.publish(cmd);
    }
}

void LoadParameters(ros::NodeHandle& nh_private) {
    nh_private.param("/traj_topic", traj_topic_, std::string("/trajectory_cmd"));
    ROS_INFO("traj_topic: %s", traj_topic_.c_str());
    nh_private.param("/sparse_waypoints_topic", sparse_waypoints_topic_, std::string("/ugv/sparse_waypoints"));
    ROS_INFO("sparse_waypoints_topic: %s", sparse_waypoints_topic_.c_str());
    nh_private.param("/odom_topic", odom_topic_, std::string("/odom"));
    ROS_INFO("odom_topic: %s", odom_topic_.c_str());
    nh_private.param("/cmd_topic", cmd_topic_, std::string("/cmd_vel"));
    ROS_INFO("cmd_topic: %s", cmd_topic_.c_str());
    nh_private.param("/waypoint_sample_dt", sample_dt_, 0.3);
    ROS_INFO("sample_dt: %f", sample_dt_);
    nh_private.param("/desired_velocity", desired_vel_, 2.2);
    nh_private.param("/desired_angular_velocity", desired_ang_vel_, 1.2);
    ROS_INFO("desired_ang_vel: %f", desired_ang_vel_);
    nh_private.param("/reach_goal_distance", reach_goal_distance_, 0.3);
    ROS_INFO("reach_goal_distance: %f", reach_goal_distance_);
    nh_private.param("/velocity_lowerbound", vel_lower_bound_, 0.5);
    ROS_INFO("vel_lower_bound: %f", vel_lower_bound_);
    nh_private.param("/angular_velocity_lowerbound", ang_vel_lower_bound_, 0.4);
    ROS_INFO("ang_vel_lower_bound: %f", ang_vel_lower_bound_);
    nh_private.param("/Kp_angle", Kp_a_, 5.0);
    ROS_INFO("Kp_a: %f", Kp_a_);
    nh_private.param("/Ki_angle", Ki_a_, 0.0);
    ROS_INFO("Ki_a: %f", Ki_a_);
    nh_private.param("/Kd_angle", Kd_a_, 0.2);
    ROS_INFO("Kd_a: %f", Kd_a_);
    nh_private.param("/Kp_distance", Kp_d_, 1.0);
    ROS_INFO("Kp_d: %f", Kp_d_);
    nh_private.param("/Ki_distance", Ki_d_, 1.0);
    ROS_INFO("Ki_d: %f", Ki_d_);
    nh_private.param("/Kd_distance", Kd_d_, 1.0);
    ROS_INFO("Kd_d: %f", Kd_d_);
    nh_private.param("/yaw_goal_tolerance", yaw_goal_tolerance_, 0.3);
    ROS_INFO("yaw_goal_tolerance: %f", yaw_goal_tolerance_);
    nh_private.param("/control_rate", control_rate_, 30.0);
    ROS_INFO("control_rate: %f", control_rate_);
    nh_private.param("/initial_backward_distance", initial_backward_distance_, 5.0);
    ROS_INFO("initial_backward_distance: %f", initial_backward_distance_);
    nh_private.param("/rotation_target_pos_threshold", rotation_target_pos_thresh_, 0.2);
    nh_private.param("/rotation_target_yaw_threshold", rotation_target_yaw_thresh_, 0.2);
    ROS_INFO("rotation_target_pos_threshold: %f", rotation_target_pos_thresh_);
    ROS_INFO("rotation_target_yaw_threshold: %f", rotation_target_yaw_thresh_);
    nh_private.param("/goal_update_pos_threshold", goal_update_pos_thresh_, 0.5);
    nh_private.param("/goal_update_yaw_threshold", goal_update_yaw_thresh_, 0.35);
    ROS_INFO("goal_update_pos_threshold: %f", goal_update_pos_thresh_);
    ROS_INFO("goal_update_yaw_threshold: %f", goal_update_yaw_thresh_);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "ugv_traj_exec");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");
    ROS_INFO("ugv_traj_exec node started");

    LoadParameters(nh_private);

    ROS_INFO("parameters loaded");

    traj_state_ = -1;
    path_active_ = false;
    rotation_mode_ = false;
    have_odom_ = false;
    initial_backward_done_ = false;
    have_rotation_target_ = false;
    have_goal_ = false;
    target_idx_ = 0;
    integral_a_ = 0.0;
    error_a_last_ = 0.0;
    zero_cmd_.linear.x = 0.0;
    zero_cmd_.angular.z = 0.0;

    traj_sub_ = nh.subscribe(traj_topic_, 10, &TrajCallback);
    sparse_waypoints_sub_ = nh.subscribe(sparse_waypoints_topic_, 10, &SparseWaypointsCallback);
    odom_sub_ = nh.subscribe(odom_topic_, 10, &OdomCallback);
    cmd_pub_ = nh.advertise<geometry_msgs::Twist>(cmd_topic_, 10);

    double dt = 1.0 / std::max(control_rate_, 1.0);
    control_timer_ = nh.createTimer(ros::Duration(dt), &ControlLoop);

    ros::spin();
    return 0;
}
